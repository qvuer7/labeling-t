"""The model-server wire contract — ONE shape for every detector.

Pydantic only (no fastapi, no torch), so adapters and the app both import it
without dragging in heavy deps. The framework's TransformersClient speaks this
exact shape; adding a model is a new ModelAdapter, never a new payload type.

    InferRequest   {image_url, queries[], params{}}
    InferResponse  {width, height, detections:[WireDetection]}
    WireDetection  {bbox:[x1,y1,x2,y2] abs-px, label, score}

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


class InferResponse(BaseModel):
    width: int
    height: int
    detections: list[WireDetection]
