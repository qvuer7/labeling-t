#!/usr/bin/env python
"""CUSTOM, one-off data-prep for the IPBL basketball dataset. NOT framework.

Sample N frames EQUALLY across the source dataset's games and copy them into a
new dataset as one flat group, so it becomes a single labeling project. The
frame filename still encodes the game, so traceability is preserved. Standalone
script — uses the framework (storage/layout/manifest) as a library. Delete freely.

    uv run python scripts/curate_frames.py --dataset ipbl-basketball \
        --out-dataset ipbl-basketball-1k --n 1000
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labeling_t.config import load_env  # noqa: E402
from labeling_t.layout import DatasetLayout  # noqa: E402
from labeling_t.manifest import build_manifest, load_manifest  # noqa: E402
from labeling_t.storage import open_storage  # noqa: E402

_IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def main(argv=None):
    load_env()
    p = argparse.ArgumentParser(description="custom: equal-per-game sample into a flat subset (not framework)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dataset", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--group", default="all", help="single output group name")
    p.add_argument("--base", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    src = DatasetLayout.from_env(a.dataset, base=a.base)
    storage = open_storage(src.root)
    frames_root = src.frames()

    by_game = collections.defaultdict(list)
    for u in storage.list(frames_root + "/"):
        if u.lower().endswith(_IMG):
            by_game[u[len(frames_root) + 1:].split("/")[0]].append(u)
    games = sorted(by_game)
    rng = random.Random(a.seed)

    # distribute n equally across games (remainder spread over the first games)
    base_n, rem = divmod(a.n, len(games))
    selected = []
    for i, g in enumerate(games):
        target = base_n + (1 if i < rem else 0)
        lst = by_game[g][:]
        rng.shuffle(lst)
        take = lst[: min(target, len(lst))]
        selected += take
        print(f"  {g}: {len(take)}/{len(by_game[g])}")
    print(f"sampled {len(selected)} frames equally across {len(games)} games")

    dst = DatasetLayout.from_env(a.out_dataset, base=a.base)
    out_group = dst.frames(a.group)
    for u in selected:
        storage.copy(u, f"{out_group}/{u.rsplit('/', 1)[-1]}")  # server-side copy

    prev = load_manifest(a.dataset, base=a.base, storage=storage) or {}
    build_manifest(
        a.out_dataset, base=a.base, storage=storage,
        categories=prev.get("categories"),
        source=f"datasets/{a.dataset} (equal-per-game sample, n={a.n})",
        stride=(prev.get("extraction") or {}).get("stride"), model=prev.get("model"),
    )
    print(f"copied {len(selected)} frames -> dataset {a.out_dataset} (group '{a.group}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
