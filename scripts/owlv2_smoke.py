#!/usr/bin/env python
"""SMOKE (throwaway): prove OWLv2 inference works end-to-end on the RunPod pod.

  1) direct /infer on one real frame  -> isolates model + adapter + unpad
  2) prelabel_cloud on a few frames   -> full pipeline: presign -> /infer -> S3
  3) read back + validate vs ImageLabels

Reads OWLV2_ENDPOINT + S3 creds from .env. Writes to a FRESH label prefix so it
doesn't collide with the qwen3-vl labels already on ipbl-basketball-1k. Delete
after PR-1 is verified.

    uv run python scripts/owlv2_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labeling_t.config import load_env  # noqa: E402
from labeling_t.model_client import TransformersClient  # noqa: E402
from labeling_t.models import get_spec  # noqa: E402
from labeling_t.prelabel import _raw_inference, prelabel_cloud  # noqa: E402
from labeling_t.schema import ImageLabels  # noqa: E402
from labeling_t.storage import open_storage  # noqa: E402

SRC = "s3://ml-cv-data/datasets/ipbl-basketball-1k/frames/all"
OUT = "s3://ml-cv-data/datasets/owlv2-smoke/labels/test"
CATS = ["player", "ball", "referee"]
N = 6


def main() -> int:
    load_env()
    spec = get_spec("owlv2")
    if not spec.endpoint_from_env():
        print("OWLV2_ENDPOINT not set", file=sys.stderr)
        return 1
    storage = open_storage(SRC)
    frames = [u for u in storage.list(SRC + "/") if u.lower().endswith((".jpg", ".jpeg", ".png"))][:N]
    if not frames:
        print("no source frames found", file=sys.stderr)
        return 1
    print(f"endpoint: {spec.endpoint_from_env()}")
    print(f"using {len(frames)} frames, e.g. {frames[0].rsplit('/', 1)[-1]}")

    client = TransformersClient.from_env(spec, categories=CATS)
    with client:
        # 1) direct /infer on one frame
        raw = _raw_inference(client, storage.presigned_url(frames[0]))
        print(f"\n[1/3 direct /infer] dims={raw.width}x{raw.height}, {len(raw.boxes)} boxes")
        for b in raw.boxes[:6]:
            print("     ", b)

        # 2) full pipeline -> S3 (fresh prefix, resume=False)
        n = prelabel_cloud(frames, client, OUT, storage=storage, resume=False, max_concurrency=4)
        print(f"\n[2/3 prelabel_cloud] wrote {n}/{len(frames)} labels -> {OUT}")

    # 3) read back + validate
    uris = [u for u in storage.list(OUT + "/") if u.endswith(".json")]
    total, cats = 0, set()
    for u in uris:
        lab = ImageLabels.model_validate_json(storage.read_bytes(u).decode())  # raises if invalid/out-of-bounds
        total += len(lab.detections)
        cats |= {d.category for d in lab.detections}
    print(f"\n[3/3 verify] {len(uris)} valid ImageLabels, {total} detections, categories={sorted(cats)}")
    print("\nOWLv2 INFERENCE OK ✅" if uris else "\nNO LABELS WRITTEN ❌")
    return 0 if uris else 1


if __name__ == "__main__":
    raise SystemExit(main())
