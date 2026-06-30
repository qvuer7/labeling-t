"""Label Studio adapter — import pre-labels, pull verified labels back.

LS is one sink, not the spine. The verified truth re-enters the neutral schema
via `from_label_studio` so export stays independent of LS.

Two confirmed LS footguns, both handled here:
  1. Bounding boxes must be PERCENT of image dims (0-100), not pixels.
  2. The prediction JSON must match a labeling-config whose <Label> values equal
     the category set. We GENERATE that config from the categories so they can't
     drift apart.

The pure functions (config gen, task building, parse-back) are fully testable
with no network. `import_to_label_studio` is the only piece that talks to a live
server; it's a thin wrapper around label-studio-sdk.

Coordinate flow:
    schema (abs-pixel xyxy) ──abs_to_percent──► LS task (percent xywh)
    LS export (percent xywh) ──percent_to_abs──► schema (abs-pixel xyxy)
"""

from __future__ import annotations

import os
from typing import Callable
from xml.sax.saxutils import quoteattr

from ..geometry import abs_to_percent, percent_to_abs, polygon_to_rle, rle_to_polygon
from ..schema import BBox, Detection, ImageLabels

# LS control tag per annotation kind; `control` selects which one a project uses.
_CONTROLS = {"rectangle": "RectangleLabels", "polygon": "PolygonLabels", "brush": "BrushLabels"}


def _image_ref(
    image_path: str,
    image_base_url: str | None,
    image_root: str | None,
    presign: Callable[[str], str] | None = None,
) -> str:
    """How the labeler's browser fetches the image.

    - `presign` set (cloud): image_path is a storage URI; return a presigned URL
      the browser fetches directly (e.g. a hosted LS loading frames from S3).
    - `image_base_url` set (local dev): base + path relative to image_root.
    - neither: pass the path/URL through unchanged.
    """
    if presign is not None:
        return presign(image_path)
    if not image_base_url:
        return image_path
    rel = os.path.relpath(image_path, image_root) if image_root else os.path.basename(image_path)
    return image_base_url.rstrip("/") + "/" + rel.replace(os.sep, "/")


def generate_label_config(
    categories: list[str],
    *,
    from_name: str = "label",
    to_name: str = "image",
    image_value: str = "image",
    control: str = "rectangle",
) -> str:
    """Build the LS labeling-config XML from the category set. Single source of
    truth: add a category, the config grows; names can never mismatch. `control`
    picks the region kind — "rectangle" (boxes) or "polygon" (verify SAM2 masks)."""
    tag = _CONTROLS[control]
    labels = "\n".join(f"    <Label value={quoteattr(c)}/>" for c in categories)
    return (
        "<View>\n"
        f'  <Image name={quoteattr(to_name)} value="${image_value}"/>\n'
        f"  <{tag} name={quoteattr(from_name)} toName={quoteattr(to_name)}>\n"
        f"{labels}\n"
        f"  </{tag}>\n"
        "</View>\n"
    )


def _rect_result(det: Detection, width: int, height: int, from_name: str, to_name: str) -> dict:
    pct = abs_to_percent(det.bbox, width, height)
    return {
        "type": "rectanglelabels",
        "from_name": from_name, "to_name": to_name,
        "original_width": width, "original_height": height, "image_rotation": 0,
        "value": {"rotation": 0, "x": pct["x"], "y": pct["y"],
                  "width": pct["width"], "height": pct["height"],
                  "rectanglelabels": [det.category]},
    }


def _polygon_result(det: Detection, width: int, height: int, from_name: str, to_name: str) -> dict | None:
    """A `polygonlabels` region from the detection's mask (RLE -> contour ->
    percent points). None if the detection has no mask or the mask is empty."""
    if not det.mask:
        return None
    poly = rle_to_polygon(det.mask)
    if poly is None:
        return None
    points = [[x / width * 100.0, y / height * 100.0] for x, y in poly]
    return {
        "type": "polygonlabels",
        "from_name": from_name, "to_name": to_name,
        "original_width": width, "original_height": height, "image_rotation": 0,
        "value": {"points": points, "closed": True, "polygonlabels": [det.category]},
    }


def _brush_result(det: Detection, width: int, height: int, from_name: str, to_name: str) -> dict | None:
    """A `brushlabels` region from the detection's mask — the raster mask itself,
    re-encoded into Label Studio's own RLE (NOT COCO RLE). None if no/empty mask.
    Lazy-imports pycocotools + LS's bundled brush encoder."""
    if not det.mask:
        return None
    import numpy as np
    from label_studio_sdk.converter import brush
    from pycocotools import mask as mask_utils

    m = np.ascontiguousarray(mask_utils.decode(det.mask)).astype(np.uint8) * 255
    if not m.any():
        return None
    return {
        "type": "brushlabels",
        "from_name": from_name, "to_name": to_name,
        "original_width": width, "original_height": height, "image_rotation": 0,
        "value": {"format": "rle", "rle": brush.mask2rle(m), "brushlabels": [det.category]},
    }


_RESULT_FNS = {"polygon": _polygon_result, "brush": _brush_result}


def _result_item(det, width, height, from_name, to_name, control: str = "rectangle"):
    return _RESULT_FNS.get(control, _rect_result)(det, width, height, from_name, to_name)


def to_label_studio_tasks(
    images: list[ImageLabels],
    *,
    from_name: str = "label",
    to_name: str = "image",
    image_value: str = "image",
    model_version: str = "qwen3-vl",
    image_base_url: str | None = None,
    image_root: str | None = None,
    presign: Callable[[str], str] | None = None,
    control: str = "rectangle",
) -> list[dict]:
    """neutral schema -> LS tasks with predictions (percent coords).

    presign (cloud): a fn turning each frame's storage URI into a URL the
    browser can fetch (presigned S3). image_base_url/image_root: local dev http.
    control="polygon" emits SAM2 masks as polygon regions (skips maskless dets).
    """
    tasks = []
    for img in images:
        results = [
            r for d in img.detections
            if (r := _result_item(d, img.width, img.height, from_name, to_name, control)) is not None
        ]
        scores = [d.score for d in img.detections if d.score is not None]
        prediction: dict = {"model_version": model_version, "result": results}
        if scores:
            prediction["score"] = sum(scores) / len(scores)
        tasks.append(
            {
                "data": {image_value: _image_ref(img.image_path, image_base_url, image_root, presign)},
                "predictions": [prediction],
            }
        )
    return tasks


def _det_from_result(item: dict) -> Detection | None:
    """One LS result (rectangle / polygon / brush) -> Detection, or None if it
    isn't a region we recognize / has no label. Polygons & brush strokes recover
    BOTH a mask (RLE) and the enclosing box; rectangles are box-only."""
    t = item.get("type")
    w, h = item["original_width"], item["original_height"]
    v = item["value"]

    if t == "rectanglelabels":
        labels = v.get("rectanglelabels") or []
        if not labels:
            return None
        return Detection(bbox=percent_to_abs(v["x"], v["y"], v["width"], v["height"], w, h),
                         category=labels[0])

    if t == "polygonlabels":
        labels = v.get("polygonlabels") or []
        pts = v.get("points") or []
        if not labels or len(pts) < 3:
            return None
        abs_pts = [(min(max(x / 100.0 * w, 0.0), float(w)),
                    min(max(y / 100.0 * h, 0.0), float(h))) for x, y in pts]
        xs = [p[0] for p in abs_pts]; ys = [p[1] for p in abs_pts]
        bbox = BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))
        return Detection(bbox=bbox, category=labels[0], mask=polygon_to_rle(abs_pts, w, h))

    if t == "brushlabels":
        labels = v.get("brushlabels") or []
        if not labels or not v.get("rle"):
            return None
        det = _brush_to_detection(v["rle"], labels[0], w, h)
        return det

    return None


def _brush_to_detection(rle: list[int], category: str, w: int, h: int) -> Detection | None:
    """LS brush RLE -> Detection (mask as COCO RLE + enclosing box). Lazy deps."""
    import numpy as np
    from label_studio_sdk.converter import brush
    from pycocotools import mask as mask_utils

    flat = np.array(brush.decode_rle(rle), dtype=np.uint8)
    m = (flat.reshape(h, w, 4).max(axis=2) > 127)  # any channel set -> painted
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    bbox = BBox(x1=float(xs.min()), y1=float(ys.min()),
                x2=float(xs.max() + 1), y2=float(ys.max() + 1))
    enc = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    coco = {"size": [int(s) for s in enc["size"]], "counts": enc["counts"].decode("ascii")}
    return Detection(bbox=bbox, category=category, mask=coco)


def from_label_studio(
    tasks: list[dict],
    *,
    image_value: str = "image",
    result_source: str = "annotations",
) -> list[ImageLabels]:
    """LS export -> neutral schema. Reads human-verified `annotations` by default
    (pass result_source="predictions" to read model output instead). Handles box,
    polygon, and brush regions; polygon/brush recover a mask AND a box. Each result
    carries original_width/height, so dims come straight from the export."""
    out: list[ImageLabels] = []
    for task in tasks:
        image_path = task.get("data", {}).get(image_value)
        if image_path is None:
            raise ValueError(f"task missing data.{image_value}: {task!r}")

        entries = task.get(result_source) or []
        results = entries[0].get("result", []) if entries else []

        width = height = None
        detections: list[Detection] = []
        for item in results:
            if item.get("type") not in ("rectanglelabels", "polygonlabels", "brushlabels"):
                continue
            width, height = item["original_width"], item["original_height"]
            det = _det_from_result(item)
            if det is not None:
                detections.append(det)

        if width is None or height is None:
            # Nothing verified for this image; can't recover dims. Skip, don't guess.
            continue
        out.append(ImageLabels(image_path=image_path, width=width, height=height,
                               detections=detections))
    return out


def import_to_label_studio(
    images: list[ImageLabels],
    *,
    base_url: str,
    api_key: str,
    project_title: str,
    categories: list[str],
    from_name: str = "label",
    to_name: str = "image",
    image_value: str = "image",
    model_version: str = "qwen3-vl",
    image_base_url: str | None = None,
    image_root: str | None = None,
    presign: Callable[[str], str] | None = None,
    control: str = "rectangle",
):  # pragma: no cover - thin I/O wrapper over label-studio-sdk
    """Create an LS project with the generated config and import the tasks +
    pre-annotations. Returns the created project. control="polygon" makes a SAM2
    mask-verification project (PolygonLabels).

    Not unit-tested (needs a live server). The testable logic lives in
    generate_label_config / to_label_studio_tasks above.
    """
    from label_studio_sdk import LabelStudio

    client = LabelStudio(base_url=base_url, api_key=api_key)
    project = client.projects.create(
        title=project_title,
        label_config=generate_label_config(
            categories, from_name=from_name, to_name=to_name,
            image_value=image_value, control=control,
        ),
    )
    tasks = to_label_studio_tasks(
        images,
        from_name=from_name,
        to_name=to_name,
        image_value=image_value,
        model_version=model_version,
        image_base_url=image_base_url,
        image_root=image_root,
        presign=presign,
        control=control,
    )
    client.projects.import_tasks(id=project.id, request=tasks)
    return project
