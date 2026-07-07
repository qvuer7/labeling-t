"""The model-server wire contract — ONE shape for every detector.

Pydantic only (no fastapi, no torch), so adapters and the app both import it
without dragging in heavy deps. The framework's TransformersClient speaks this
exact shape; adding a model is a new ModelAdapter, never a new payload type.

    InferRequest   {image_url, queries[], params{}}   (SAM2 box prompts ride in params)
    InferResponse  {width, height, detections:[WireDetection]}
    WireDetection  {bbox:[x1,y1,x2,y2] abs-px, label, score, mask?:RLE}

`bbox` is a flat 4-list in ABSOLUTE pixels of the ORIGINAL image; the framework
maps it to a schema BBox via _box_to_bbox(coord_space="abs").
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    image_url: str
    queries: list[str] = Field(default_factory=list)
    params: dict = Field(default_factory=dict)


class WireDetection(BaseModel):
    bbox: list[float]  # [x1, y1, x2, y2], absolute pixels, original image space
    label: str
    score: float | None = None
    # Optional segmentation mask, COCO RLE: {"size": [h, w], "counts": "<rle>"}.
    # Only the segmenter (SAM2) fills this; detectors leave it None. Stays on the
    # SAME wire shape so a box and its mask travel together (two-stage: detector
    # box -> SAM2 mask), and COCO export gets `segmentation` for free later.
    mask: dict | None = None
    # Optional keypoints, [{x, y, name, score?}] in abs pixels of the original
    # image. Only the pose stage (VitPose) fills this — same ride-along rule as
    # mask: a box and its skeleton travel together.
    keypoints: list[dict] | None = None


class InferResponse(BaseModel):
    width: int
    height: int
    detections: list[WireDetection]
