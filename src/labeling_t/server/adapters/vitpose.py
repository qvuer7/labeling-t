"""VitPose adapter — the KEYPOINTS stage (top-down pose).

VitPose is not a detector: like SAM2 it takes geometric PROMPTS (person boxes)
and returns a skeleton per box. Stage 2 of the two-stage pipeline — a detector
produces `player` boxes, VitPose fills each box's keypoints:

    detector ─► boxes ─► /infer {image_url, params:{boxes,labels,scores}} ─► VitPose
                 WireDetection{bbox, label, keypoints:[{x,y,name,score}]} ◄── per box

transformers-NATIVE (VitPoseForPoseEstimation, since 4.48 — inside the ==4.57.1
pin), plain torch: same slim model-server image, no new deps. Box prompts ride
`params` exactly like SAM2's; only WireDetection gained an optional `keypoints`.

Default weights are the single-dataset COCO-17 variant
(usyd-community/vitpose-base-simple). The MoE multi-dataset variants
(vitpose-base et al.) additionally need a `dataset_index` — out of scope until
a non-COCO skeleton is actually wanted.

Keypoint NAMES: we emit our canonical COCO-17 names ("left_knee", …) whenever
the model returns exactly 17 points; otherwise the model config's id2label
(lowercased) or "kp_<i>". Canonical-first keeps every player file in a dataset
uniformly named regardless of which checkpoint produced it.
"""

from __future__ import annotations

import io

from ..contract import InferResponse, WireDetection

# The COCO keypoint order — the convention VitPose COCO checkpoints are trained on.
COCO_17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


def _xyxy_to_xywh(boxes: list[list[float]]) -> list[list[float]]:
    """Framework xyxy boxes -> the COCO xywh the VitPose processor expects."""
    return [[float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1)]
            for x1, y1, x2, y2 in boxes]


def _point_names(n: int, id2label: dict | None) -> list[str]:
    """Names for an n-point skeleton: canonical COCO-17 when n == 17, else the
    model's own id2label (lowercased), else positional kp_<i>."""
    if n == len(COCO_17_NAMES):
        return list(COCO_17_NAMES)
    if id2label:
        return [str(id2label.get(i, id2label.get(str(i), f"kp_{i}"))).lower() for i in range(n)]
    return [f"kp_{i}" for i in range(n)]


def _to_wire_keypoints(xy, scores, names: list[str]) -> list[dict]:
    """One pose result -> wire keypoints [{x, y, name, score}] in image pixels.
    `xy` is an (n, 2) array-like, `scores` an (n,) array-like or None."""
    out = []
    for i, (x, y) in enumerate(xy):
        kp = {"x": float(x), "y": float(y), "name": names[i]}
        if scores is not None:
            kp["score"] = max(0.0, min(float(scores[i]), 1.0))
        out.append(kp)
    return out


class VitPoseAdapter:
    """usyd-community/vitpose-* via transformers. torch imported lazily."""

    def __init__(self, hf_model: str) -> None:
        self.hf_model = hf_model
        self.ready = False
        self._model = None
        self._processor = None
        self._device = "cpu"

    def load(self) -> None:  # pragma: no cover - needs torch + weights (GPU pod)
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.hf_model)
        self._model = VitPoseForPoseEstimation.from_pretrained(self.hf_model).to(self._device).eval()
        self.ready = True

    def detect(self, image_url: str, queries: list[str], params: dict) -> InferResponse:  # pragma: no cover - needs torch
        import httpx
        import torch
        from PIL import Image

        boxes = params.get("boxes") or []
        labels = params.get("labels") or [None] * len(boxes)
        scores = params.get("scores") or [None] * len(boxes)

        resp = httpx.get(image_url, timeout=60.0)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        w, h = image.size
        if not boxes:
            return InferResponse(width=w, height=h, detections=[])

        # processor wants per-image COCO xywh boxes: [[box, box, ...]] for our 1 image
        xywh = _xyxy_to_xywh(boxes)
        inputs = self._processor(image, boxes=[xywh], return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        # post-process rescales heatmap peaks back to ORIGINAL image pixels;
        # [0] = our single image -> one result dict per input box, in order.
        poses = self._processor.post_process_pose_estimation(outputs, boxes=[xywh])[0]

        id2label = getattr(self._model.config, "id2label", None)
        dets: list[WireDetection] = []
        for box, label, score, pose in zip(boxes, labels, scores, poses):
            xy = pose["keypoints"].cpu().numpy()
            kp_scores = pose.get("scores")
            kp_scores = kp_scores.cpu().numpy() if kp_scores is not None else None
            names = _point_names(len(xy), id2label)
            dets.append(WireDetection(
                bbox=[float(v) for v in box], label=label or "object", score=score,
                keypoints=_to_wire_keypoints(xy, kp_scores, names),
            ))
        return InferResponse(width=w, height=h, detections=dets)
