#!/usr/bin/env python
"""CUSTOM, one-off data-prep: the type-2 scoreboard KEYPOINT dataset. NOT framework.

Samples N frames per game (evenly spaced through each game -> time/rig variety)
from the raw parent pool into datasets/ipbl-scoreboard-kp/ as one flat group,
and seeds each frame's label with one `scoreboard` detection carrying three
keypoints (home / away / timer) at the KNOWN type-2 region centers — so in
Label Studio the annotator drags 3 pre-placed points instead of clicking from
scratch. On frames from a different rig the seeds are simply wrong and get
dragged into place; that IS the labeling task.

    uv run python scripts/make_scoreboard_kp_dataset.py

Then:  labeling-t import-ls-cloud --dataset ipbl-scoreboard-kp --group all \
           --project "ipbl scoreboard keypoints (type2 mix)" \
           --categories home,away,timer --keypoints --json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labeling_t.config import load_env  # noqa: E402
from labeling_t.layout import DatasetLayout  # noqa: E402
from labeling_t.manifest import build_manifest  # noqa: E402
from labeling_t.schema import BBox, Detection, ImageLabels, Keypoint  # noqa: E402
from labeling_t.storage import open_storage  # noqa: E402

# --- CONFIG ---------------------------------------------------------------
SRC_DATASET = "ipbl-basketball"          # raw parent pool (28 games)
OUT_DATASET = "ipbl-scoreboard-kp"
GAMES = [
    "dogs_vs_camels_02",
    "cats_vs_wolves_02",
    "lynx_vs_dolphins_02",
    "owls_vs_iguana_02",
    "owls_vs_lynx_01",
]
PER_GAME = 40                            # 5 x 40 = 200 frames
WIDTH, HEIGHT = 1280, 720                # verified for all five games

# type-2 scoreboard regions (x1, y1, x2, y2) -> seed keypoint = region center
REGIONS = {
    "home":  (240, 605, 290, 640),
    "away":  (352, 606, 405, 640),
    "timer": (70, 570, 113, 593),
}
# ---------------------------------------------------------------------------


def _center(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def main() -> None:
    load_env()
    src = DatasetLayout.from_env(SRC_DATASET)
    out = DatasetLayout.from_env(OUT_DATASET)
    storage = open_storage(src.root)

    # one detection: the scoreboard area (union of the regions) carrying the seeds
    xs = [v for r in REGIONS.values() for v in (r[0], r[2])]
    ys = [v for r in REGIONS.values() for v in (r[1], r[3])]
    seed_det = Detection(
        bbox=BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)),
        category="scoreboard",
        keypoints=[Keypoint(x=_center(*r)[0], y=_center(*r)[1], name=name)
                   for name, r in REGIONS.items()],
    )

    total = 0
    for game in GAMES:
        frames = sorted(u for u in storage.list(src.frames(game) + "/") if u.endswith(".jpg"))
        if len(frames) < PER_GAME:
            raise SystemExit(f"{game}: only {len(frames)} frames, wanted {PER_GAME}")
        step = len(frames) / PER_GAME    # evenly spaced -> spread over game time
        picks = [frames[int(i * step)] for i in range(PER_GAME)]
        for uri in picks:
            stem = uri.rsplit("/", 1)[-1][:-4]
            frame_dest = f"{out.frames('all')}/{stem}.jpg"
            storage.copy(uri, frame_dest)
            labels = ImageLabels(image_path=frame_dest, width=WIDTH, height=HEIGHT,
                                 detections=[seed_det])
            storage.write_text(f"{out.labels('all')}/{stem}.json", labels.model_dump_json())
            total += 1
        print(f"{game}: {PER_GAME} frames sampled (of {len(frames)})")

    build_manifest(OUT_DATASET, categories=list(REGIONS))
    print(f"done: {total} frames + seeded labels -> {out.root}")


if __name__ == "__main__":
    main()
