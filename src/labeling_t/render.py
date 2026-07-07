"""Render labels onto frames — the visual-verification primitive.

Agents (and humans) judge label quality fastest by LOOKING: net-in-box,
hallucinated boxes, purple-lighting failures were all caught visually. This
replaces the hand-rolled PIL snippet of every such check with one command:
boxes + captions (category, score, text) + optional 25%-alpha mask overlays,
written as local PNGs regardless of where the labels live.

Boxes/captions need only PIL (base install). Mask overlays decode COCO RLE via
pycocotools, lazy-imported — without the [integrations] extra, rendering a
masked set fails loudly naming the extra, box-only sets still work.
"""

from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from .labelset import _label_files
from .schema import ImageLabels

# Stable, high-contrast palette; a category always gets the same color within
# and across runs (index by sorted-name hash, not first-seen order).
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
]
_MASK_ALPHA = 64  # ≈25% of 255


def _color(category: str) -> tuple[int, int, int]:
    return _PALETTE[sum(category.encode()) % len(_PALETTE)]


def _decode_rle(rle: dict):
    """COCO RLE -> HxW uint8 array. Lazy pycocotools; the error names the fix."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:  # pragma: no cover - dev env has the extra
        raise ImportError(
            "mask rendering needs pycocotools — install the extra: "
            "pip install 'labeling-t[integrations]'"
        ) from exc
    counts = rle["counts"]
    return mask_utils.decode({"size": rle["size"],
                              "counts": counts.encode() if isinstance(counts, str) else counts})


def render_labels(labels: ImageLabels, image_bytes: bytes,
                  *, skeleton: list[list[str]] | None = None) -> bytes:
    """One frame + its labels -> annotated PNG bytes (pure, no I/O).

    Masks first (25%-alpha tint), then box outlines, then a caption of
    `category [score]` with `Detection.text` on a second line when present,
    then keypoints (white-ringed dots; `skeleton` = optional list of
    [name_a, name_b] edges drawn between named points when both exist)."""
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    for d in labels.detections:
        if d.mask is not None:
            color = _color(d.category)
            m = _decode_rle(d.mask)
            tint = Image.new("RGB", im.size, color)
            alpha = Image.fromarray((m * _MASK_ALPHA).astype("uint8"), mode="L")
            im.paste(tint, (0, 0), alpha)
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for d in labels.detections:
        color = _color(d.category)
        box = (d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2)
        draw.rectangle(box, outline=color, width=2)
        caption = d.category if d.score is None else f"{d.category} {d.score:.2f}"
        lines = [caption] + ([d.text] if d.text else [])
        y = d.bbox.y1
        for line in lines:
            bb = draw.textbbox((d.bbox.x1, y), line, font=font)
            draw.rectangle(bb, fill=color)
            draw.text((d.bbox.x1, y), line, fill=(0, 0, 0), font=font)
            y = bb[3]
    for d in labels.detections:
        if not d.keypoints:
            continue
        color = _color(d.category)
        pts = {k.name: (k.x, k.y) for k in d.keypoints}
        for a, b in skeleton or []:  # edges under the dots
            if a in pts and b in pts:
                draw.line([pts[a], pts[b]], fill=color, width=2)
        r = 3
        for k in d.keypoints:
            draw.ellipse([k.x - r, k.y - r, k.x + r, k.y + r],
                         fill=color, outline=(255, 255, 255))
    out = io.BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


def render_set(
    prefix: str,
    *,
    storage,
    out_dir: str,
    stems: set[str] | None = None,
    sample: int | None = None,
    seed: int = 0,
    skeleton: list[list[str]] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Render a label set's frames to local PNGs (<out_dir>/<stem>.png).

    `stems` restricts, then `sample` draws a DETERMINISTIC random subset
    (same seed = same stems — re-render after a fix shows the same frames).
    Frames come from each label's image_path via the same storage. Per-stem
    problems (missing frame, bad RLE) land in `failures`, never abort the run;
    a missing pycocotools is a config error and does abort. Returns
    {rendered, out, stems, failures: [{stem, error}]}."""
    files = _label_files(storage, prefix)
    selected = sorted(files)
    if stems is not None:
        selected = [s for s in selected if s in stems]
    if sample is not None and sample < len(selected):
        selected = sorted(random.Random(seed).sample(selected, sample))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []
    rendered = 0
    for i, stem in enumerate(selected, 1):
        try:
            labels = ImageLabels.model_validate_json(storage.read_bytes(files[stem]).decode())
            png = render_labels(labels, storage.read_bytes(labels.image_path),
                                skeleton=skeleton)
            (out / f"{stem}.png").write_bytes(png)
            rendered += 1
        except ImportError:
            raise  # missing [integrations] — fix the env, not the file
        except Exception as exc:  # noqa: BLE001 — a bad file is a finding, not a crash
            failures.append({"stem": stem, "error": f"{type(exc).__name__}: {exc}"})
        if on_progress is not None:
            on_progress(i, len(selected))
    return {"rendered": rendered, "out": str(out), "stems": selected, "failures": failures}
