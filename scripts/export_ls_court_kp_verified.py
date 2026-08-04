#!/usr/bin/env python3
"""THROWAWAY: pull verified court keypoints from LS project 32 -> YOLO-pose txt.

Mirrors the seed-arc-v1/labels/ format: one line per frame,
`0 cx cy w h` + 29 x `(x y v)` normalized; v=2 for every human-verified point,
`0 0 0` for absent. bbox = enclosing box of the visible points. LS keypoint
coords are percent, so /100 is already normalized — image dims never needed.

Usage:
    uv run python scripts/export_ls_court_kp_verified.py \
        --project 32 --out data/court-kp-verified
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

NAMES = [
    "cc_center", "centerline_L", "centerline_R", "cc_L", "cc_R",
    "A_corner_L", "A_corner_R", "A_lane_base_L", "A_lane_base_R",
    "A_lane_ft_L", "A_lane_ft_R", "A_ft_center", "A_corner3_L", "A_corner3_R",
    "A_arcjunc_L", "A_arcjunc_R", "A_arc_apex",
    "B_corner_L", "B_corner_R", "B_lane_base_L", "B_lane_base_R",
    "B_lane_ft_L", "B_lane_ft_R", "B_ft_center", "B_corner3_L", "B_corner3_R",
    "B_arcjunc_L", "B_arcjunc_R", "B_arc_apex",
]


def fetch_export(base: str, key: str, project: int) -> list[dict]:
    url = f"{base}/api/projects/{project}/export?exportType=JSON&download_all_tasks=true"
    req = urllib.request.Request(url, headers={"Authorization": f"Token {key}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = os.environ["LS_URL"].rstrip("/")
    key = os.environ["LS_API_KEY"]
    tasks = fetch_export(base, key, a.project)

    out = Path(a.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    (out / "raw" / "ls_export.json").write_text(json.dumps(tasks))

    name2id = {n: i for i, n in enumerate(NAMES)}
    written = skipped_unann = dupes = unknown = 0
    vis_counts: Counter[int] = Counter()

    for task in tasks:
        anns = [x for x in task.get("annotations", []) if not x.get("was_cancelled")]
        if not anns:
            skipped_unann += 1
            continue
        results = anns[0].get("result", [])
        pts: dict[int, tuple[float, float]] = {}
        for item in results:
            if item.get("type") != "keypointlabels":
                continue
            labels = item["value"].get("keypointlabels") or []
            if not labels or labels[0] not in name2id:
                unknown += 1
                continue
            kid = name2id[labels[0]]
            if kid in pts:
                dupes += 1
            pts[kid] = (item["value"]["x"] / 100.0, item["value"]["y"] / 100.0)

        stem = Path(task["data"]["image"].split("?")[0]).stem
        if not pts:
            (out / "labels" / f"{stem}.txt").write_text("")
            written += 1
            continue
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        parts = [f"0 {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}"]
        for kid in range(len(NAMES)):
            p = pts.get(kid)
            parts.append(f"{p[0]:.6f} {p[1]:.6f} 2" if p else "0 0 0")
        (out / "labels" / f"{stem}.txt").write_text(" ".join(parts) + "\n")
        vis_counts[len(pts)] += 1
        written += 1

    manifest = {
        "source": f"LS project {a.project} (human-verified)",
        "total": written,
        "skipped_unannotated": skipped_unann,
        "duplicate_points_overwritten": dupes,
        "unknown_labels_skipped": unknown,
        "points_per_frame": dict(sorted(vis_counts.items())),
        "names": NAMES,
        "note": "v=2 human-verified; 0 0 0 absent. bbox = enclosing box of visible points.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
