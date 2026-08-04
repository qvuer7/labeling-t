#!/usr/bin/env python3
"""Densify court-anchor labels: derive every in-frame court keypoint via homography.

The step between human anchor clicks and keypoint-detector training:

    clicked anchors (>=4) ──DLT──► H (court meters -> image px)
    all 29 model points ──H──► image; DROP out-of-frame (YOLO v=0)
    in-frame: v=1 (occluded: inside a player/referee mask from labels-v2)
              v=2 (visible) — the detector trains on BOTH; occlusion is
              recorded, not excluded.
    human-clicked coords are kept verbatim (locally more accurate than the fit);
    derived coords fill everything else.

Neutral schema has no visibility field, so it rides on keypoint `score`:
    1.0 = visible (v=2) · 0.5 = occluded (v=1) · absent = out of frame (v=0)

Frames with <4 named anchors are skipped (no H) and listed in the summary CSV.

Usage:
    uv run python scripts/derive_court_points.py \
        --anchors data/verified-court-anchors \
        --masks data/labels-v2-local \
        --court /home/andrii/Projects/vsoccer/court_reg_refs/court_keypoints_29.json \
        --out data/labels-court-kp-dense --summary data/court_kp_dense_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

_PREFIX = re.compile(r"^\d+_")
_MARGIN = 2.0  # px: treat borderline-edge projections as out of frame


def _dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalized DLT homography src->dst, both (N,2), N>=4."""
    def norm(pts):
        c = pts.mean(0)
        s = np.sqrt(2) / max(np.mean(np.linalg.norm(pts - c, axis=1)), 1e-9)
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
        return (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :2], T

    a, Ta = norm(src)
    b, Tb = norm(dst)
    rows = []
    for (x, y), (X, Y) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, X * x, X * y, X])
        rows.append([0, 0, 0, -x, -y, -1, Y * x, Y * y, Y])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    H = np.linalg.inv(Tb) @ vt[-1].reshape(3, 3) @ Ta
    return H / H[2, 2]


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = (H @ np.c_[pts, np.ones(len(pts))].T).T
    return p[:, :2] / p[:, 2:3]


def _occluder_mask(doc: dict):
    """Union of player/referee RLE masks from a labels-v2 file -> HxW bool, or None."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None
    m = None
    for d in doc.get("detections", []):
        if d.get("category") not in ("player", "referee") or not d.get("mask"):
            continue
        rle = dict(d["mask"])
        if isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
        dec = mask_utils.decode(rle).astype(bool)
        m = dec if m is None else (m | dec)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--masks", required=True, help="labels-v2 dir (player/referee RLE masks)")
    ap.add_argument("--court", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--skip", default="",
                    help="comma-separated point names to never derive (features "
                         "not painted on this floor, e.g. IPBL has no center circle)")
    a = ap.parse_args()
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}

    court = json.loads(Path(a.court).read_text())
    names = {v["name"]: np.array(v["court_xy_m"], float) for v in court.values()}
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summ = []
    for f in sorted(Path(a.anchors).glob("*.json")):
        doc = json.loads(f.read_text())
        det = doc["detections"][0]
        w, h = doc.get("width", 1280), doc.get("height", 720)
        clicked = {}
        for kp in det.get("keypoints", []):
            n = _PREFIX.sub("", kp["name"])
            if n in names and n not in clicked:  # first click wins on dupes
                clicked[n] = (kp["x"], kp["y"])
        if len(clicked) < 4:
            summ.append({"frame": f.stem, "anchors": len(clicked), "status": "skipped-<4",
                         "in_frame": 0, "visible": 0, "occluded": 0})
            continue

        H = _dlt(np.array([names[n] for n in clicked]),      # court -> image
                 np.array([clicked[n] for n in clicked]))
        occl = None
        mf = Path(a.masks) / f.name
        if mf.exists():
            occl = _occluder_mask(json.loads(mf.read_text()))

        kps, vis_n, occ_n = [], 0, 0
        for n, cxy in names.items():
            if n in skip and n not in clicked:
                continue  # feature not painted on this floor -> v=0, never derived
            if n in clicked:
                x, y = clicked[n]
            else:
                x, y = _project(H, cxy[None, :])[0]
            if not (_MARGIN <= x <= w - _MARGIN and _MARGIN <= y <= h - _MARGIN):
                continue  # v=0: out of frame, never trained
            occluded = bool(occl is not None and occl[min(int(y), h - 1), min(int(x), w - 1)])
            kps.append({"x": float(x), "y": float(y), "name": n,
                        "score": 0.5 if occluded else 1.0})
            occ_n += occluded
            vis_n += not occluded
        # Degenerate-H guard: few / near-collinear anchors can yield an H that
        # collapses many distinct court points onto one image spot. If any two
        # DIFFERENT points land within 5 px, distrust every derived point and
        # keep only the human clicks for this frame.
        collapsed = any(
            abs(p["x"] - q["x"]) < 5 and abs(p["y"] - q["y"]) < 5
            for i, p in enumerate(kps) for q in kps[i + 1:]
        )
        if collapsed:
            kps = [p for p in kps if _PREFIX.sub("", p["name"]) in clicked]
            vis_n, occ_n = len(kps), 0
            if not kps:
                summ.append({"frame": f.stem, "anchors": len(clicked),
                             "status": "degenerate-H", "in_frame": 0,
                             "visible": 0, "occluded": 0})
                continue
        xs = [p["x"] for p in kps]; ys = [p["y"] for p in kps]
        out = {"image_path": doc["image_path"], "width": w, "height": h,
               "schema_version": "2",
               "detections": [{"bbox": {"x1": max(0.0, min(xs) - 20), "y1": max(0.0, min(ys) - 20),
                                        "x2": min(float(w), max(xs) + 20), "y2": min(float(h), max(ys) + 20)},
                               "category": "court", "score": None,
                               "source": det.get("source", "human") + "+H-derived",
                               "keypoints": kps}]}
        (out_dir / f.name).write_text(json.dumps(out))
        summ.append({"frame": f.stem, "anchors": len(clicked), "status": "ok",
                     "in_frame": len(kps), "visible": vis_n, "occluded": occ_n})

    with open(a.summary, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["frame", "anchors", "status",
                                            "in_frame", "visible", "occluded"])
        wr.writeheader()
        wr.writerows(summ)
    ok = [s for s in summ if s["status"] == "ok"]
    print(f"frames: {len(summ)} | densified: {len(ok)} | skipped(<4 anchors): {len(summ) - len(ok)}")
    if ok:
        print(f"avg in-frame points: {sum(s['in_frame'] for s in ok) / len(ok):.1f} / 29 "
              f"(visible {sum(s['visible'] for s in ok)}, occluded {sum(s['occluded'] for s in ok)})")
    print(f"labels -> {out_dir} | summary -> {a.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
