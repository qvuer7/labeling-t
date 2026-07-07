"""Neutral label schema — the owned contract every integration hangs off.

This is the spine of the whole tool. Models write INTO this; Label Studio,
COCO, YOLO, FiftyOne all read FROM it via adapters. Nothing downstream is the
canonical representation; this is.

v0 scoped to Detection (bounding boxes) only, images only. Extensions since:
masks (2026-07, schema_version 1) and keypoints (2026-07, schema_version 2) —
both shipped as OPTIONAL FIELDS ON Detection, not as sibling annotation kinds,
because both are box-DERIVED in this pipeline: masks come from box-prompted
SAM2, keypoints from top-down pose (a box in, that box's skeleton out). The
sibling-collection design (COCO/FiftyOne pattern: `masks: list[Mask]`,
`keypoints: list[Keypoints]` next to `detections`) remains the right move iff
a PROMPTLESS producer ever lands (bottom-up pose, image-level court corners);
until such data exists, don't build it.

Still deliberately absent: Classification, track_id, video. Same rule.

Coordinate convention (CANONICAL, do not deviate):

    absolute-pixel xyxy  ->  (x1, y1) top-left, (x2, y2) bottom-right, in pixels

    (0,0)─────────────► x
      │   (x1,y1)
      │      ┌───────┐
      │      │  box  │
      │      └───────┘
      │           (x2,y2)
      ▼
      y

Every other coordinate system (model-normalized, Label-Studio percent, COCO
xywh) converts to/from this ONE reference in geometry.py. Picking pixels here is
what lets `sv.Detections` (also abs-pixel) be a zero-cost export boundary.

`extra="forbid"` on every model is intentional: it makes the schema refuse
unknown fields, so the contract can't quietly accumulate per-case fields. That
field discipline — not "CV labels are a finite set" — is what keeps this from
sprawling the way prior attempts did.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Shared strict config: no unknown fields, validate on assignment.
_STRICT = ConfigDict(extra="forbid", validate_assignment=True)


class BBox(BaseModel):
    """Axis-aligned bounding box in absolute pixels, xyxy."""

    model_config = _STRICT

    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_ordering(self) -> "BBox":
        if self.x2 < self.x1:
            raise ValueError(f"x2 ({self.x2}) must be >= x1 ({self.x1})")
        if self.y2 < self.y1:
            raise ValueError(f"y2 ({self.y2}) must be >= y1 ({self.y1})")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class Keypoint(BaseModel):
    """One named point of a detection's keypoint set, absolute pixels.

    NAMED ("left_knee"), not index-positioned: the file stays self-describing
    without an external skeleton definition. Adapters map names to COCO's
    index order at the export boundary, where such conventions belong.
    `visible` follows COCO's spirit: True = visible (v=2), False = labeled but
    occluded (v=1), None = unknown/not stated."""

    model_config = _STRICT

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    name: str = Field(min_length=1)
    visible: bool | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class Detection(BaseModel):
    """One detected object: a box, a category, and where it came from.

    `mask` is the optional segmentation that goes WITH this box — our masks come
    from box-prompted SAM2 (one mask per box), so they pair naturally on the same
    Detection rather than living in a separate list. Stored as COCO RLE
    ({"size": [h, w], "counts": str}); absent for box-only labels. (This is the
    schema.py extension plan's step done minimally: masks ride on Detection
    because they're box-derived; a free-standing Mask kind would only be needed
    for promptless segmentation, which we don't do.)

    `text` follows the same pattern: it is the transcription OF this box — a
    second-stage OCR pass (transcribe.py) crops the detection and asks a VLM to
    read it, so the text is box-derived and rides the Detection like a mask does.
    None = never transcribed; "" = transcribed, nothing legible (the distinction
    is what lets a resumed run skip already-attempted crops).

    `keypoints` completes the trio: the skeleton OF this box, from a top-down
    pose pass (keypoints.py) that prompts the pose model with this bbox. Same
    resume contract as text: None = never attempted; [] = attempted, nothing
    found.
    """

    model_config = _STRICT

    bbox: BBox
    category: str = Field(min_length=1)
    # Pre-label confidence in [0,1]. None once a human has verified it.
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    # Provenance: which model/run produced this (e.g. "locate-anything-3b").
    source: str | None = None
    # Optional COCO RLE segmentation for this box (SAM2). None = box-only.
    mask: dict | None = None
    # Optional transcription of this box (second-stage OCR). None = not
    # attempted; "" = attempted, nothing legible.
    text: str | None = None
    # Optional keypoints of this box (top-down pose). None = not attempted;
    # [] = attempted, nothing found.
    keypoints: list[Keypoint] | None = None


class ImageLabels(BaseModel):
    """All labels for a single image, plus the image dimensions.

    width/height are REQUIRED, not optional: every coordinate conversion
    (normalized -> pixel -> percent -> COCO) needs them. Carrying them here is
    what stops geometry round-trips from passing on synthetic data and breaking
    on real frames.
    """

    model_config = _STRICT

    # Contract version this file was written under; every dump writes it.
    # "2" = Detection.keypoints exists (extra="forbid" means pre-2 readers
    # reject files that carry keypoints — hence the bump). Loaders accept any
    # older version; the ON-DISK value is the truth for provenance (stats reads
    # it from the raw JSON, where a pre-versioning file counts as "absent" —
    # this pydantic default is only what the CURRENT code writes).
    schema_version: str = "2"
    image_path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections: list[Detection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_annotations_within_image(self) -> "ImageLabels":
        """Per-kind bounds check: every coordinate-bearing part of a detection
        (bbox, keypoints) must sit inside the image."""
        for d in self.detections:
            if d.bbox.x2 > self.width or d.bbox.y2 > self.height:
                raise ValueError(
                    f"detection bbox {d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2} "
                    f"exceeds image bounds {self.width}x{self.height}"
                )
            for k in d.keypoints or []:
                if k.x > self.width or k.y > self.height:
                    raise ValueError(
                        f"keypoint {k.name!r} ({k.x}, {k.y}) exceeds image "
                        f"bounds {self.width}x{self.height}"
                    )
        return self
