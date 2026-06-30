"""Neutral label schema — the owned contract every integration hangs off.

This is the spine of the whole tool. Models write INTO this; Label Studio,
COCO, YOLO, FiftyOne all read FROM it via adapters. Nothing downstream is the
canonical representation; this is.

v0 scope: Detection (bounding boxes) only, images only.
Deliberately NOT in v0: Mask, Keypoints, Classification, track_id, video. Those
are real CV label types and will be added when a dataset actually needs them
(Stage-2 SAM masks, tracking). Defining them now would be building for data that
doesn't exist yet — the over-generalization this project exists to avoid.

HORIZONTAL EXTENSION PLAN (do this when the SECOND annotation type lands, not
before — it's a mechanical, non-breaking refactor):

  1. Extract the metadata shared by every annotation kind into a base, so a new
     kind (and any future shared field like track_id) is a single-point add:

         class Annotation(BaseModel):     # category / score / source / track_id
             category: str
             score: float | None = None
             source: str | None = None
         class Detection(Annotation): bbox: BBox
         class Mask(Annotation):      polygon: list[list[float]]   # or RLE
         class Keypoints(Annotation): points: list[Keypoint]

  2. Add sibling collections on ImageLabels (COCO/FiftyOne pattern) — new field,
     new type, nothing existing changes:

         detections: list[Detection] = []
         masks:      list[Mask]      = []
         keypoints:  list[Keypoints] = []

  3. Generalize _check_boxes_within_image into a per-kind bounds check.

Until then Detection stays flat: one type, no base, no speculative structure.

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


class Detection(BaseModel):
    """One detected object: a box, a category, and where it came from.

    `mask` is the optional segmentation that goes WITH this box — our masks come
    from box-prompted SAM2 (one mask per box), so they pair naturally on the same
    Detection rather than living in a separate list. Stored as COCO RLE
    ({"size": [h, w], "counts": str}); absent for box-only labels. (This is the
    schema.py extension plan's step done minimally: masks ride on Detection
    because they're box-derived; a free-standing Mask kind would only be needed
    for promptless segmentation, which we don't do.)
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


class ImageLabels(BaseModel):
    """All labels for a single image, plus the image dimensions.

    width/height are REQUIRED, not optional: every coordinate conversion
    (normalized -> pixel -> percent -> COCO) needs them. Carrying them here is
    what stops geometry round-trips from passing on synthetic data and breaking
    on real frames.
    """

    model_config = _STRICT

    image_path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detections: list[Detection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_boxes_within_image(self) -> "ImageLabels":
        for d in self.detections:
            if d.bbox.x2 > self.width or d.bbox.y2 > self.height:
                raise ValueError(
                    f"detection bbox {d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2} "
                    f"exceeds image bounds {self.width}x{self.height}"
                )
        return self
