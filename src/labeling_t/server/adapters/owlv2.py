"""OWLv2 open-vocab detector adapter.

THE OWLv2 TRAP (the reason PR-1 carries a non-square test): Owlv2Processor pads
the image to a SQUARE (side = max(W,H)) before inference, and
post_process_object_detection rescales boxes to `target_sizes` as if the image
were that square. On a non-square frame (our 16:9 video), passing the original
(H,W) yields boxes scaled wrong. The fix: post-process against the SQUARE size,
then clamp to the original frame and DROP boxes that fall in the padding region
(bottom/right). OWL-ViT doesn't have this; OWLv2 does.

    padded square (max(W,H))              original frame
    ┌───────────────┐                     ┌───────────────┐
    │ real image    │ ◄── boxes here ──►  │ keep + clamp  │
    │               │                     └───────────────┘
    ├───────────────┤  ◄── boxes here ──►  DROP (padding, not real pixels)
    │ padding       │
    └───────────────┘

`_finalize` is a pure function (no torch) so the unpad/clamp/filter/label-map
logic is unit-tested without a GPU; `detect()` is the thin torch wrapper.
"""

from __future__ import annotations

import io

from ..contract import InferResponse, WireDetection


def _finalize(
    boxes: list[list[float]],
    scores: list[float],
    label_ids: list[int],
    queries: list[str],
    orig_w: int,
    orig_h: int,
    threshold: float,
) -> list[WireDetection]:
    """Padded-square-pixel boxes -> original-frame WireDetections.

    Drops sub-threshold boxes, boxes whose top-left sits in the padding region
    (not real pixels), and degenerate boxes after clamping. Maps each label id
    to its query string.
    """
    out: list[WireDetection] = []
    for box, score, lid in zip(boxes, scores, label_ids):
        if score < threshold:
            continue
        if lid < 0 or lid >= len(queries):
            continue  # defensive: post_process should never index out of range
        x1, y1, x2, y2 = box
        if x1 >= orig_w or y1 >= orig_h:
            continue  # top-left in the padding region -> phantom detection, drop it
        cx1, cy1 = max(0.0, float(x1)), max(0.0, float(y1))
        cx2, cy2 = min(float(orig_w), float(x2)), min(float(orig_h), float(y2))
        if cx2 <= cx1 or cy2 <= cy1:
            continue  # nothing left after clamping
        out.append(WireDetection(bbox=[cx1, cy1, cx2, cy2], label=queries[lid], score=float(score)))
    return out


class Owlv2Adapter:
    """google/owlv2-* via transformers. torch imported lazily in load()/detect()."""

    def __init__(self, hf_model: str) -> None:
        self.hf_model = hf_model
        self.ready = False
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self) -> None:  # pragma: no cover - needs torch + weights (GPU pod)
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Owlv2Processor.from_pretrained(self.hf_model)
        self._model = Owlv2ForObjectDetection.from_pretrained(self.hf_model).to(self._device).eval()
        self.ready = True

    def detect(self, image_url: str, queries: list[str], params: dict) -> InferResponse:  # pragma: no cover - needs torch
        import httpx
        import torch
        from PIL import Image

        threshold = float(params.get("box_threshold", params.get("threshold", 0.1)))
        resp = httpx.get(image_url, timeout=60.0)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        w, h = image.size

        inputs = self._processor(text=[queries], images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        side = max(w, h)  # OWLv2 pads to a square; post-process against THAT, then unpad
        target_sizes = torch.tensor([[side, side]], device=self._device)
        res = self._processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        dets = _finalize(
            res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist(),
            queries, w, h, threshold,
        )
        return InferResponse(width=w, height=h, detections=dets)
