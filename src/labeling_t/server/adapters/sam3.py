"""SAM3 (Segment Anything 3) adapter — text-prompted detector AND segmenter.

SAM3 does Promptable Concept Segmentation (PCS): a short noun phrase in (e.g.
"yellow school bus"), instance masks + boxes + confidence scores for EVERY
matching object out. That collapses the two-stage pipeline (detector boxes ->
SAM2 masks) into one pass: /infer with `queries` returns masked detections
directly, no box prompts needed.

Like LocateAnything, one query per forward pass — but the vision embeddings are
computed ONCE and reused across queries (the model exposes get_vision_features
for exactly this), so extra categories cost only the cheap text+DETR half:

    queries=[player, ball]  ──►  vision embeds (once)
                                 per-query text ──► forward ──► post_process
                                 masks+boxes+scores (abs px) ──► _to_wire
                                 masks ride as COCO RLE (same as SAM2)

NOTE: needs transformers>=5 (Sam3Model landed 2025-11), which conflicts with the
default image's ==4.57.1 pin (LocateAnything's vendored remote code). So sam3
serves from its OWN image variant — ModelSpec.image_tag="sam3", built with the
Dockerfile's TRANSFORMERS_VERSION build-arg. Weights are gated on HF (HF_TOKEN).

`_to_wire` is pure (no torch) so the box/score/mask plumbing is unit-tested
without a GPU; `load()`/`detect()` are the thin torch wrappers (lazy import).
"""

from __future__ import annotations

import io

from ..contract import InferResponse, WireDetection
from .sam2 import _encode_rle


def _to_wire(
    boxes: list[list[float]],
    scores: list[float | None],
    masks: list[dict | None],
    label: str,
    w: int,
    h: int,
) -> list[WireDetection]:
    """Post-processed abs-px boxes/scores/RLE-masks for one query -> wire
    detections. Clamps to the frame, tolerates flipped corners, drops boxes
    that are degenerate after clamping (their mask goes with them)."""
    out: list[WireDetection] = []
    for box, score, mask in zip(boxes, scores, masks):
        x1, y1, x2, y2 = (float(v) for v in box)
        cx1, cx2 = sorted((x1, x2))
        cy1, cy2 = sorted((y1, y2))
        cx1, cy1 = max(0.0, cx1), max(0.0, cy1)
        cx2, cy2 = min(float(w), cx2), min(float(h), cy2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue  # nothing left after clamping -> drop
        out.append(WireDetection(
            bbox=[cx1, cy1, cx2, cy2], label=label,
            score=None if score is None else float(score), mask=mask,
        ))
    return out


class Sam3Adapter:
    """facebook/sam3 via transformers (native Sam3Model, no trust_remote_code).
    torch imports lazily in load()/detect()."""

    def __init__(self, hf_model: str) -> None:
        self.hf_model = hf_model
        self.ready = False
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self) -> None:  # pragma: no cover - needs torch + weights (GPU pod)
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Sam3Processor.from_pretrained(self.hf_model)
        self._model = Sam3Model.from_pretrained(self.hf_model).to(self._device).eval()
        self.ready = True

    def detect(self, image_url: str, queries: list[str], params: dict) -> InferResponse:  # pragma: no cover - needs torch
        import httpx
        import torch
        from PIL import Image

        resp = httpx.get(image_url, timeout=60.0)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        w, h = image.size

        threshold = float(params.get("threshold", 0.5))
        mask_threshold = float(params.get("mask_threshold", 0.5))

        # Vision embeddings once; each query then pays only the text+DETR half.
        img_inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            vision_embeds = self._model.get_vision_features(pixel_values=img_inputs.pixel_values)

        dets: list[WireDetection] = []
        for q in queries or ["object"]:
            text_inputs = self._processor(text=q, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(vision_embeds=vision_embeds, **text_inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs, threshold=threshold, mask_threshold=mask_threshold,
                target_sizes=img_inputs.get("original_sizes").tolist(),
            )[0]
            boxes = [[float(v) for v in b] for b in results["boxes"].cpu().tolist()]
            scores = [float(s) for s in results["scores"].cpu().tolist()]
            masks = [_encode_rle(m.cpu().numpy()) for m in results["masks"]]
            dets.extend(_to_wire(boxes, scores, masks, q, w, h))
        return InferResponse(width=w, height=h, detections=dets)
