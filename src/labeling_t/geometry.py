"""Coordinate conversions — the single bug-prone zone, isolated on purpose.

ALL coordinate math lives here. No adapter does its own division by image
dimensions. If percent/pixel/normalized confusion is going to bite, it bites in
one tested file instead of scattered across three adapters.

Conversion hub (everything goes through abs-pixel xyxy, see schema.py):

    model output (normalized)                 Label Studio (percent, xywh)
            │  normalized_to_abs                     ▲  abs_to_percent
            ▼                                         │
        ┌──────────────────── abs-pixel xyxy ────────────────────┐
            │  abs_to_coco_xywh
            ▼
        COCO (abs-pixel xywh)

Normalized scale note: VLM grounding models disagree on normalized range.
LocateAnything emits [0,1] floats; Qwen-style models emit 0-1000 integers.
`normalized_to_abs` takes a `scale` so both are one call (default 1.0 = [0,1]).
The exact scale for LocateAnything is confirmed in the T0 spike.
"""

from __future__ import annotations

from .schema import BBox


def normalized_to_abs(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    scale: float = 1.0,
) -> BBox:
    """Model-normalized xyxy (in [0, scale]) -> absolute-pixel BBox.

    scale=1.0 for [0,1] models (LocateAnything); scale=1000 for 0-1000 models.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image dims must be positive, got {width}x{height}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    return BBox(
        x1=x1 / scale * width,
        y1=y1 / scale * height,
        x2=x2 / scale * width,
        y2=y2 / scale * height,
    )


def abs_to_normalized(
    box: BBox, width: int, height: int, scale: float = 1.0
) -> tuple[float, float, float, float]:
    """Absolute-pixel BBox -> normalized xyxy in [0, scale]. Inverse of
    `normalized_to_abs`; exists mainly so round-trips are testable."""
    if width <= 0 or height <= 0:
        raise ValueError(f"image dims must be positive, got {width}x{height}")
    return (
        box.x1 / width * scale,
        box.y1 / height * scale,
        box.x2 / width * scale,
        box.y2 / height * scale,
    )


def abs_to_percent(box: BBox, width: int, height: int) -> dict[str, float]:
    """Absolute-pixel BBox -> Label Studio rectangle value.

    LS expects top-left x/y plus width/height, each as a PERCENT (0-100) of the
    image dimension — NOT pixels. This is the #1 LS import footgun.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image dims must be positive, got {width}x{height}")
    return {
        "x": box.x1 / width * 100.0,
        "y": box.y1 / height * 100.0,
        "width": box.width / width * 100.0,
        "height": box.height / height * 100.0,
    }


def percent_to_abs(
    x: float, y: float, w: float, h: float, width: int, height: int
) -> BBox:
    """Label Studio percent rectangle (x,y,w,h in 0-100) -> absolute-pixel
    BBox. Used by `from_label_studio` to pull verified labels back."""
    if width <= 0 or height <= 0:
        raise ValueError(f"image dims must be positive, got {width}x{height}")
    x1 = x / 100.0 * width
    y1 = y / 100.0 * height
    x2 = x1 + w / 100.0 * width
    y2 = y1 + h / 100.0 * height
    # A labeler can draw a box to the exact edge or slightly off-canvas; a box at
    # 100% also lands a float-epsilon past the bound (e.g. 1280.0000000002).
    # Clamp to the frame so the pulled-back BBox is always in-bounds.
    def _clamp(v: float, hi: int) -> float:
        return max(0.0, min(v, float(hi)))
    return BBox(x1=_clamp(x1, width), y1=_clamp(y1, height),
                x2=_clamp(x2, width), y2=_clamp(y2, height))


def abs_to_coco_xywh(box: BBox) -> list[float]:
    """Absolute-pixel BBox -> COCO [x, y, width, height] (top-left + size, abs
    pixels). No image dims needed — COCO uses absolute pixels like we do."""
    return [box.x1, box.y1, box.width, box.height]


def rle_to_polygon(rle: dict, *, simplify: float = 0.004) -> list[tuple[float, float]] | None:
    """COCO RLE mask -> simplified outer-contour polygon, abs-pixel (x, y) points.

    Used to turn SAM2's raster masks into editable polygons (Label Studio verifies
    polygons far more easily than brush masks, and polygons round-trip to COCO
    `segmentation`). Returns the largest external contour, Douglas-Peucker-
    simplified (`simplify` = epsilon as a fraction of the contour perimeter); None
    for an empty/degenerate mask. Lazy-imports pycocotools + cv2 — only needed when
    masks are in play, so the base install stays light."""
    import cv2
    import numpy as np
    from pycocotools import mask as mask_utils

    m = np.ascontiguousarray(mask_utils.decode(rle)).astype(np.uint8)
    if m.sum() == 0:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, simplify * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return [(float(x), float(y)) for x, y in approx]


def polygon_to_rle(points: list[tuple[float, float]], width: int, height: int) -> dict:
    """Abs-pixel polygon points -> COCO RLE mask (the polygon filled on a WxH
    canvas). Inverse direction of `rle_to_polygon`, for pulling verified polygons
    back into the schema. Lazy-imports cv2 + pycocotools."""
    import cv2
    import numpy as np
    from pycocotools import mask as mask_utils

    canvas = np.zeros((height, width), dtype=np.uint8)
    pts = np.array([[round(x), round(y)] for x, y in points], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], 1)
    enc = mask_utils.encode(np.asfortranarray(canvas))
    return {"size": [int(s) for s in enc["size"]],
            "counts": enc["counts"].decode("ascii")}
