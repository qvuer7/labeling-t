"""Second-stage pose — fill `Detection.keypoints` on existing labels.

The third enrichment stage (segment.py fills `mask`, transcribe.py fills
`text`; this fills `keypoints`): read a label set, send each frame's matching
boxes to the pose model (VitPose on our model-server) as box prompts, and
write the returned skeletons back onto the same Detections.

    labels/<stem>.json ──boxes──► TransformersClient.keypoints (VitPose)
        ──one skeleton per box──► Detection.keypoints ──► same JSON, rewritten

Resume is PER-DETECTION, keyed on `keypoints is None` — a resumed run only
prompts for boxes that still lack a skeleton ([] = attempted, counts as done).
The frame travels as a presigned URL (the GPU fetches it; nothing is
downloaded here). Failures go to `keypoints_failures.jsonl` and do not stop
the run. Structure deliberately mirrors segment.py — the stages stay
diff-able against each other.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .prelabel import _WRITE_BACKOFF, _WRITE_RETRIES
from .schema import ImageLabels

FAILURES_NAME = "keypoints_failures.jsonl"


def _targets(labels: ImageLabels, categories: set[str] | None) -> list:
    """The detections this run still has to pose."""
    return [d for d in labels.detections
            if d.keypoints is None and (categories is None or d.category in categories)]


def _fill_keypoints(labels: ImageLabels, image_url: str, client, *,
                    categories: set[str] | None) -> int:
    """Prompt the pose model with every skeleton-less matching box, in place.
    Returns how many detections gained keypoints (an empty skeleton counts —
    it marks the box as attempted, per the resume contract)."""
    todo = _targets(labels, categories)
    if not todo:
        return 0
    out = client.keypoints(
        image_url,
        [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in todo],
        labels=[d.category for d in todo],
        scores=[d.score for d in todo],
    )
    if len(out) != len(todo):
        raise ValueError(f"pose model returned {len(out)} detections for {len(todo)} boxes")
    filled = 0
    for det, wire in zip(todo, out):  # server returns one detection per box, in order
        kps = wire.get("keypoints")
        if kps is None:
            continue
        # a heatmap peak can land a float-epsilon outside the frame; clamp so
        # the schema's bounds validator never rejects the whole file over it
        det.keypoints = [
            {**k, "x": max(0.0, min(float(k["x"]), float(labels.width))),
             "y": max(0.0, min(float(k["y"]), float(labels.height)))}
            for k in kps
        ]
        filled += 1
    return filled


def keypoints_cloud(
    labels_prefix: str,
    client,
    *,
    storage,
    categories: list[str] | None = None,
    to_prefix: str | None = None,
    stems: set[str] | None = None,
    max_concurrency: int = 1,
    resume: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Fill a cloud label set's keypoints from its boxes.

    In place by default; `to_prefix` writes enriched copies instead (source
    untouched; resume reads the copy first). `categories=None` poses every
    box — in practice pass the person-like category (e.g. player). `stems`
    restricts the run (sample-first workflow). Default concurrency is 1: the
    pose model runs on the transformers backend (one model, one GPU, not
    reentrant). Returns the number of label files enriched this run.
    """
    cats = set(categories) if categories else None
    src = labels_prefix.rstrip("/")
    out = to_prefix.rstrip("/") if to_prefix else src
    keys = sorted(k for k in storage.list(src + "/") if k.endswith(".json"))
    if stems is not None:
        keys = [k for k in keys if k.rsplit("/", 1)[-1][:-5] in stems]
    done_set = set(storage.list(out + "/")) if (resume and to_prefix) else set()
    fails: list[str] = []
    lock = threading.Lock()

    def work(key: str) -> int:
        name = key.rsplit("/", 1)[-1]
        dest = f"{out}/{name}"
        try:
            read_from = dest if dest in done_set else key
            labels = ImageLabels.model_validate_json(storage.read_bytes(read_from))
            if resume and not _targets(labels, cats):
                return 0
            filled = _fill_keypoints(labels, storage.presigned_url(labels.image_path),
                                     client, categories=cats)
            if not filled:
                return 0
            payload = labels.model_dump_json()
            for attempt in range(_WRITE_RETRIES):
                try:
                    storage.write_text(dest, payload)
                    break
                except Exception:  # noqa: BLE001
                    if attempt == _WRITE_RETRIES - 1:
                        raise
                    time.sleep(_WRITE_BACKOFF * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - resilience is the point
            with lock:
                fails.append(json.dumps({"labels": key, "error": f"{type(exc).__name__}: {exc}"}))
            return 0
        return 1

    enriched = done = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for fut in as_completed(pool.submit(work, k) for k in keys):
            enriched += fut.result()
            done += 1
            if on_progress is not None:
                on_progress(done, len(keys))
    if fails:
        storage.write_text(f"{out}/{FAILURES_NAME}", "\n".join(fails) + "\n")
        print(f"{len(fails)} files failed -> {out}/{FAILURES_NAME}", file=sys.stderr)
    return enriched
