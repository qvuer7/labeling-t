"""ModelAdapter — the per-model contract on the server side.

One adapter per model. `detect()` takes the uniform request fields and returns
the uniform response; the adapter owns everything model-specific (input format,
post-processing, coord normalization), so the wire contract stays identical
across models and the framework never learns per-model quirks.

    detect(image_url, queries, params) -> InferResponse
      real adapter: fetch image_url, run the model, normalize to abs-px xyxy
"""

from __future__ import annotations

from typing import Protocol

from ..contract import InferResponse, WireDetection


class ModelAdapter(Protocol):
    ready: bool

    def load(self) -> None:
        """Load weights (heavy; lazy torch import inside). Sets ready=True."""
        ...

    def detect(self, image_url: str, queries: list[str], params: dict) -> InferResponse:
        ...


class StubAdapter:
    """No-model, no-torch adapter: one fixed box per query at canned dims. Lets
    the seam (and CI) run without a GPU. MODEL=stub selects it."""

    def __init__(self) -> None:
        self.ready = True

    def load(self) -> None:  # nothing to load
        self.ready = True

    def detect(self, image_url: str, queries: list[str], params: dict) -> InferResponse:
        dets = [
            WireDetection(bbox=[10.0, 10.0, 110.0, 110.0], label=q, score=0.5)
            for q in (queries or ["object"])
        ]
        return InferResponse(width=640, height=480, detections=dets)
