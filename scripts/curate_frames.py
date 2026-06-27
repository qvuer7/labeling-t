#!/usr/bin/env python
"""CUSTOM, one-off data-prep for the IPBL basketball dataset. NOT framework.

This is dataset-specific curation (drop dark/flat/corrupted frames, sample a
subset) — it is deliberately a standalone script, not a `labeling-t` command,
because the heuristics are mine and not reusable. It just uses the framework's
storage/layout/manifest as a library. Delete it freely.

    # see the quality distribution first (no writes):
    uv run python scripts/curate_frames.py --dataset ipbl-basketball --n 1000 --dry-run
    # then create the curated subset:
    uv run python scripts/curate_frames.py --dataset ipbl-basketball --out-dataset ipbl-basketball-1k --n 1000
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageStat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labeling_t.config import load_env  # noqa: E402
from labeling_t.layout import DatasetLayout  # noqa: E402
from labeling_t.manifest import build_manifest, load_manifest  # noqa: E402
from labeling_t.storage import open_storage  # noqa: E402

_IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _group(frames_root, uri):
    return uri[len(frames_root) + 1:].split("/")[0]


def _score(storage, uri):
    """(brightness, contrast) on a 64x64 grayscale, or None if it won't decode."""
    try:
        im = Image.open(io.BytesIO(storage.read_bytes(uri))).convert("L").resize((64, 64))
        st = ImageStat.Stat(im)
        return st.mean[0], st.stddev[0]
    except Exception:
        return None


def _pct(values, ps=(5, 25, 50, 75, 95)):
    if not values:
        return {}
    s = sorted(values)
    return {p: round(s[min(len(s) - 1, int(p / 100 * len(s)))], 1) for p in ps}


def main(argv=None):
    load_env()
    p = argparse.ArgumentParser(description="custom frame curation (not framework)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dataset", default=None)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--base", default=None)
    p.add_argument("--min-brightness", type=float, default=30.0)
    p.add_argument("--min-contrast", type=float, default=18.0)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    src = DatasetLayout.from_env(a.dataset, base=a.base)
    storage = open_storage(src.root)
    frames_root = src.frames()
    frames = [u for u in storage.list(frames_root + "/") if u.lower().endswith(_IMG)]

    scores, corrupt = {}, []
    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        for uri, sc in zip(frames, pool.map(lambda u: _score(storage, u), frames)):
            corrupt.append(uri) if sc is None else scores.__setitem__(uri, sc)

    good = [u for u, (b, c) in scores.items() if b >= a.min_brightness and c >= a.min_contrast]
    dark = [u for u, (b, c) in scores.items() if b < a.min_brightness]
    flat = [u for u, (b, c) in scores.items() if b >= a.min_brightness and c < a.min_contrast]
    print(json.dumps({
        "total": len(frames), "good": len(good), "too_dark": len(dark),
        "too_flat": len(flat), "corrupt": len(corrupt),
        "brightness_pct": _pct([b for b, _ in scores.values()]),
        "contrast_pct": _pct([c for _, c in scores.values()]),
    }, indent=2))

    if a.dry_run:
        return 0
    if not a.out_dataset:
        print("--out-dataset required (or use --dry-run)", file=sys.stderr)
        return 1

    by_group = collections.defaultdict(list)
    for u in good:
        by_group[_group(frames_root, u)].append(u)
    rng = random.Random(0)
    selected = []
    for lst in by_group.values():
        rng.shuffle(lst)
        selected += lst[: round(a.n * len(lst) / max(1, len(good)))]
    rng.shuffle(selected)
    selected = selected[: a.n]

    dst = DatasetLayout.from_env(a.out_dataset, base=a.base)
    for u in selected:
        storage.copy(u, f"{dst.frames(_group(frames_root, u))}/{u.rsplit('/', 1)[-1]}")
    prev = load_manifest(a.dataset, base=a.base, storage=storage) or {}
    build_manifest(a.out_dataset, base=a.base, storage=storage,
                   categories=prev.get("categories"),
                   source=f"datasets/{a.dataset}/frames (curated)",
                   stride=(prev.get("extraction") or {}).get("stride"), model=prev.get("model"))
    print(f"copied {len(selected)} curated frames -> dataset {a.out_dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
