#!/usr/bin/env python3
"""QA court-anchor keypoint labels against the known court geometry.

For each frame with >=5 clicked anchors, fit an image->court homography (DLT,
numpy only) from the anchors' known court positions (court_keypoints_29.json,
meters) and report each point's reprojection residual IN METERS — a bad click
shows up as a large residual. Frames with exactly 4 anchors fit exactly (no
residual signal) and are reported as "minimal"; <4 can't solve H at all.

Also runs structural checks that need no homography:
  - duplicate numbered anchor labels in one frame (protocol: max 1 each)
  - far/near sanity: `*_left` (far sideline) should sit HIGHER in the image
    (smaller y) than its `*_right` twin for a normal broadcast camera
  - points outside the frame bounds

Usage:
    uv run python scripts/qa_court_anchors.py \
        --labels data/verified-court-anchors \
        --court /home/andrii/Projects/vsoccer/court_reg_refs/court_keypoints_29.json \
        --out data/court_anchor_qa.csv [--threshold-m 0.35]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

_PREFIX = re.compile(r"^\d+_")  # LS labels carry the schema id prefix: "05_A_..."


def _dlt_homography(img: np.ndarray, court: np.ndarray) -> np.ndarray:
    """Normalized DLT: image px -> court meters. img/court are (N,2), N>=4."""
    def norm(pts):
        c = pts.mean(0)
        s = np.sqrt(2) / max(np.mean(np.linalg.norm(pts - c, axis=1)), 1e-9)
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        return (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :2], T

    a, Ta = norm(img)
    b, Tb = norm(court)
    rows = []
    for (x, y), (X, Y) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, X * x, X * y, X])
        rows.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tb) @ Hn @ Ta
    return H / H[2, 2]


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = (H @ np.c_[pts, np.ones(len(pts))].T).T
    return p[:, :2] / p[:, 2:3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="dir of neutral-schema label JSONs")
    ap.add_argument("--court", required=True, help="court_keypoints_29.json (meters)")
    ap.add_argument("--out", required=True, help="CSV report path")
    ap.add_argument("--threshold-m", type=float, default=0.35,
                    help="flag points with residual above this (meters)")
    a = ap.parse_args()

    model = {v["name"]: tuple(v["court_xy_m"]) for v in json.loads(Path(a.court).read_text()).values()}
    rows, frame_summ = [], []
    for f in sorted(Path(a.labels).glob("*.json")):
        doc = json.loads(f.read_text())
        det = doc["detections"][0]
        src = det.get("source", "?")
        w, h = doc.get("width", 1280), doc.get("height", 720)
        named, arcs, issues = [], 0, []
        seen: dict[str, int] = {}
        for kp in det.get("keypoints", []):
            name = _PREFIX.sub("", kp["name"])
            if name.startswith("arc_"):
                arcs += 1
                continue
            if name not in model:
                issues.append(f"unknown-label:{kp['name']}")
                continue
            seen[name] = seen.get(name, 0) + 1
            if not (0 <= kp["x"] <= w and 0 <= kp["y"] <= h):
                issues.append(f"out-of-bounds:{name}")
            named.append((name, kp["x"], kp["y"]))
        issues += [f"duplicate:{n}" for n, c in seen.items() if c > 1]

        # far/near sanity: the far (*_left) twin should be higher in the image
        pos = dict((n, (x, y)) for n, x, y in named)
        for ln, (lx, ly) in pos.items():
            if ln.endswith("_left"):
                rn = ln[:-5] + "_right"
                if rn in pos and ly > pos[rn][1]:
                    issues.append(f"far-below-near:{ln}")

        status, rms = "minimal", None
        if len(named) >= 5:
            img = np.array([[x, y] for _, x, y in named])
            court = np.array([model[n] for n, _, _ in named])
            H = _dlt_homography(img, court)
            res = np.linalg.norm(_project(H, img) - court, axis=1)
            rms = float(np.sqrt(np.mean(res ** 2)))
            status = "ok" if rms <= a.threshold_m else "high-rms"
            for (n, x, y), r in zip(named, res):
                rows.append({"frame": f.stem, "source": src, "point": n,
                             "residual_m": round(float(r), 3),
                             "flag": "OUTLIER" if r > a.threshold_m else ""})
        elif len(named) == 4:
            status = "minimal-4pt"
        else:
            status = f"insufficient-{len(named)}pt"
        frame_summ.append({"frame": f.stem, "source": src, "anchors": len(named),
                           "arc_pts": arcs, "status": status,
                           "rms_m": None if rms is None else round(rms, 3),
                           "issues": ";".join(issues)})

    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["frame", "source", "anchors", "arc_pts",
                                            "status", "rms_m", "issues"])
        wr.writeheader()
        wr.writerows(frame_summ)
    pt_out = Path(a.out).with_suffix(".points.csv")
    with open(pt_out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["frame", "source", "point", "residual_m", "flag"])
        wr.writeheader()
        wr.writerows(rows)

    # console summary
    from collections import Counter, defaultdict
    by_src_status = defaultdict(Counter)
    res_by_src = defaultdict(list)
    for s in frame_summ:
        by_src_status[s["source"]][s["status"]] += 1
    for r in rows:
        res_by_src[r["source"]].append(r["residual_m"])
    print(f"frames: {len(frame_summ)} | point residuals computed: {len(rows)}")
    for src, cnt in sorted(by_src_status.items()):
        rs = res_by_src.get(src, [])
        med = float(np.median(rs)) if rs else None
        outl = sum(1 for r in rs if r > a.threshold_m)
        print(f"  {src}: {dict(cnt)} | median residual: {med} m | outlier points: {outl}/{len(rs)}")
    bad = [s for s in frame_summ if s["status"] == "high-rms" or s["issues"]]
    print(f"frames needing eyes: {len(bad)}")
    for s in bad[:15]:
        print(f"    {s['frame']} [{s['source']}] {s['status']} rms={s['rms_m']} {s['issues']}")
    print(f"reports: {a.out} + {pt_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
