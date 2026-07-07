#!/usr/bin/env python
"""CUSTOM, one-off data-prep: the court-keypoints dataset. NOT framework.

Reuses the 200 frames sampled for ipbl-scoreboard-kp (5 type-2 games, evenly
spaced) as a NEW dataset for 29-point court-geometry labeling. Each frame gets
one full-frame `court` detection with NO keypoints — the labeler places only
the points visible in that frame (absence = not visible; YOLO-pose treats a
missing point as v=0).

Point schema (user's 29-keypoint court): indices 0-4 midcourt, 5-16 the NEAR
end (the one filling the frame), 17-28 the FAR end. LS label values are
index-prefixed so they stay unique and preserve YOLO keypoint order.
15/27 (arc_junction_right) and 16/28 (arc_top) were truncated in the source
message and inferred from the left/right pairing — edit the LS config if wrong.

    uv run python scripts/make_court_kp_dataset.py

Then:  labeling-t import-ls-cloud --dataset ipbl-court-kp --group all \
           --project "ipbl court keypoints (29pt)" \
           --categories "$(uv run python scripts/make_court_kp_dataset.py --print-points)" \
           --keypoints --json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labeling_t.config import load_env  # noqa: E402
from labeling_t.layout import DatasetLayout  # noqa: E402
from labeling_t.manifest import build_manifest  # noqa: E402
from labeling_t.schema import BBox, Detection, ImageLabels  # noqa: E402
from labeling_t.storage import open_storage  # noqa: E402

SRC_DATASET = "ipbl-scoreboard-kp"       # same 200 frames, different labeling task
OUT_DATASET = "ipbl-court-kp"
WIDTH, HEIGHT = 1280, 720

_MID = [
    "center_circle_center", "centerline_left", "centerline_right",
    "center_circle_left", "center_circle_right",
]
_END = [
    "corner_left", "corner_right", "lane_base_left", "lane_base_right",
    "lane_ft_left", "lane_ft_right", "ft_center", "corner3_left",
    "corner3_right", "arc_junction_left", "arc_junction_right", "arc_top",
]
# 0-4 midcourt, 5-16 near end, 17-28 far end -> 29 index-prefixed label values
POINTS = (
    [f"{i:02d}_{n}" for i, n in enumerate(_MID)]
    + [f"{i + 5:02d}_{n}" for i, n in enumerate(_END)]
    + [f"{i + 17:02d}_{n}_far" for i, n in enumerate(_END)]
)
assert len(POINTS) == 29


def main() -> None:
    if "--print-points" in sys.argv:
        print(",".join(POINTS))
        return
    load_env()
    src = DatasetLayout.from_env(SRC_DATASET)
    out = DatasetLayout.from_env(OUT_DATASET)
    storage = open_storage(src.root)

    frames = sorted(u for u in storage.list(src.frames("all") + "/") if u.endswith(".jpg"))
    if not frames:
        raise SystemExit(f"no frames under {src.frames('all')} — run make_scoreboard_kp_dataset.py first")
    court = Detection(bbox=BBox(x1=0, y1=0, x2=WIDTH, y2=HEIGHT), category="court")
    for uri in frames:
        stem = uri.rsplit("/", 1)[-1][:-4]
        frame_dest = f"{out.frames('all')}/{stem}.jpg"
        storage.copy(uri, frame_dest)
        labels = ImageLabels(image_path=frame_dest, width=WIDTH, height=HEIGHT,
                             detections=[court])
        storage.write_text(f"{out.labels('all')}/{stem}.json", labels.model_dump_json())

    build_manifest(OUT_DATASET, categories=["court"])
    print(f"done: {len(frames)} frames + empty court labels -> {out.root}")


if __name__ == "__main__":
    main()
