#!/usr/bin/env python3
"""THROWAWAY: merge two S3 label sets of the same dataset/group into a new set.

Per stem: detections(A) + detections(B) -> dst, schema-validated. A is the base
(its width/height/image_path win); stems missing in B just copy A. Built for
labels + labels-sam3-court-final -> labels-v2 (2026-07-20); delete when the
framework grows a real merge stage.

Usage:
    uv run python scripts/merge_label_sets.py \
        --dataset ipbl-basketball-seg --group all \
        --a labels --b labels-sam3-court-final --dst labels-v2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from labeling_t.config import load_env  # noqa: E402
from labeling_t.schema import ImageLabels  # noqa: E402

EP = ["--endpoint-url", "https://fra1.digitaloceanspaces.com"]
BUCKET = "ml-cv-data"


def sync(src: str, dst: str) -> None:
    subprocess.run(["aws", "s3", "sync", src, dst, *EP, "--only-show-errors"], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--a", required=True, help="base label set (kept as-is)")
    ap.add_argument("--b", required=True, help="set whose detections are appended")
    ap.add_argument("--dst", required=True, help="new set name to write")
    # legacy flat sets (labels/*.json, imported with --group "") have no group
    # segment; "" skips it for that side. dst always uses --group.
    ap.add_argument("--a-group", default=None, help='group for set A ("" = flat legacy set)')
    ap.add_argument("--b-group", default=None, help='group for set B ("" = flat legacy set)')
    args = ap.parse_args()
    load_env()

    def prefix(name: str, group: str | None) -> str:
        g = args.group if group is None else group
        seg = f"/{g}" if g else ""
        return f"s3://{BUCKET}/datasets/{args.dataset}/{name}{seg}/"

    base = f"s3://{BUCKET}/datasets/{args.dataset}"
    with tempfile.TemporaryDirectory() as td:
        da, db, dd = Path(td, "a"), Path(td, "b"), Path(td, "dst")
        sync(prefix(args.a, args.a_group), str(da))
        sync(prefix(args.b, args.b_group), str(db))
        dd.mkdir()

        merged = appended = 0
        for fa in sorted(da.glob("*.json")):
            la = ImageLabels.model_validate_json(fa.read_text())
            fb = db / fa.name
            if fb.exists():
                lb = ImageLabels.model_validate_json(fb.read_text())
                if (lb.width, lb.height) != (la.width, la.height):
                    raise SystemExit(f"dim mismatch on {fa.stem}: "
                                     f"a={la.width}x{la.height} b={lb.width}x{lb.height}")
                la.detections.extend(lb.detections)
                appended += len(lb.detections)
            (dd / fa.name).write_text(la.model_dump_json())
            merged += 1

        if merged == 0:
            raise SystemExit("no files matched set A — check --a/--a-group (nothing written)")
        sync(str(dd), prefix(args.dst, None))
        print(f"merged {merged} files, appended {appended} detections "
              f"-> {prefix(args.dst, None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
