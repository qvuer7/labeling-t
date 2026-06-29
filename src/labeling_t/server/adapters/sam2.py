"""SAM2 (Segment Anything 2) adapter — the SEGMENTER stage.

SAM2 is not a detector: it has no text understanding and emits no boxes. It takes
geometric PROMPTS (here, boxes) and returns a MASK per prompt. So it's stage 2 of
a two-stage pipeline — a detector (OWLv2/LocateAnything/Qwen3-VL) produces boxes,
then SAM2 turns each box into a mask:

    detector ─► boxes ─► /infer {image_url, params:{boxes,labels,scores}} ─► SAM2
                          WireDetection{bbox, label, mask:RLE} ◄── one mask per box

We use transformers' NATIVE Sam2 (Sam2Model + Sam2Processor), NOT facebookresearch
/sam2 — the native path is plain torch (no custom CUDA `_C` extension to compile),
so it rides the same slim model-server image as the other adapters. The box prompts
arrive in `params` (the contract's free-form dict), so no detector adapter and no
wire shape changes — only WireDetection gained an optional `mask`.

Masks are returned as COCO RLE ({"size":[h,w], "counts": str}) so they're compact on
the wire and COCO-export-ready. `_input_boxes` (pure) is unit-tested; `_encode_rle`
needs pycocotools; `detect()`/`load()` are the thin torch wrappers (lazy import).
"""

from __future__ import annotations

import io

from ..contract import InferResponse, WireDetection


def _input_boxes(boxes: list[list[float]]) -> list[list[list[float]]]:
    """Framework boxes ([[x1,y1,x2,y2], ...]) -> SAM2's nested prompt shape
    [batch][num_boxes][4]. One image per call, so a single batch entry."""
    return [[list(map(float, b)) for b in boxes]]


def _encode_rle(mask) -> dict:  # pragma: no cover - needs pycocotools (in the [models] image)
    """Boolean HxW mask -> COCO RLE with a JSON-safe (str) `counts`."""
    import numpy as np
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    return {"size": [int(s) for s in rle["size"]],
            "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts}


class Sam2Adapter:
    """facebook/sam2.1-* via transformers. torch imported lazily in load()/detect()."""

    def __init__(self, hf_model: str) -> None:
        self.hf_model = hf_model
        self.ready = False
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self) -> None:  # pragma: no cover - needs torch + weights (GPU pod)
        import torch
        from transformers import Sam2Model, Sam2Processor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Sam2Processor.from_pretrained(self.hf_model)
        self._model = Sam2Model.from_pretrained(self.hf_model).to(self._device).eval()
        self.ready = True

    def detect(self, image_url: str, queries: list[str], params: dict) -> InferResponse:  # pragma: no cover - needs torch
        import httpx
        import torch
        from PIL import Image

        boxes = params.get("boxes") or []
        # carry the prompt's category/score through onto the mask (SAM2 itself is
        # class-agnostic); default when a detector didn't supply them.
        labels = params.get("labels") or [None] * len(boxes)
        scores = params.get("scores") or [None] * len(boxes)

        resp = httpx.get(image_url, timeout=60.0)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        w, h = image.size
        if not boxes:
            return InferResponse(width=w, height=h, detections=[])

        inputs = self._processor(
            images=image, input_boxes=_input_boxes(boxes), return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)
        # post_process_masks rescales to the ORIGINAL frame; [0] = our single image,
        # giving a [num_boxes, 1, H, W] boolean tensor (multimask_output=False -> 1).
        masks = self._processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], binarize=True
        )[0]

        dets: list[WireDetection] = []
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            m = masks[i]
            if m.ndim == 3:
                m = m[0]
            rle = _encode_rle(m.cpu().numpy().astype(bool))
            dets.append(WireDetection(
                bbox=[float(v) for v in box], label=label or "object",
                score=score, mask=rle,
            ))
        return InferResponse(width=w, height=h, detections=dets)
