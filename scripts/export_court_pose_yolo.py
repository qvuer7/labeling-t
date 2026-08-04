#!/usr/bin/env python3
"""Export dense court-keypoint labels to a YOLO-pose dataset (kpt_shape [29,3]).

Per frame: one `court` object — bbox = the label's court bbox, then all 29
schema points in id order as (x, y, v) normalized to image size:
    v=2 visible (score 1.0) · v=1 occluded (score 0.5) · v=0 absent (x=y=0)
Absent covers out-of-frame points AND deliberately untrained features (center
circle / arc points excluded upstream by derive_court_points.py --skip).

Split is BY GAME (frame stems share a `<game>_NNN_NNNNN` prefix) so near-
duplicate frames of one game never straddle train/val. dataset.yaml includes
flip_idx: horizontal flip swaps basket A ids (5-16) with basket B ids (17-28);
far/near (`_left`/`_right`) suffixes are unaffected by a horizontal flip.

Usage:
    uv run python scripts/export_court_pose_yolo.py \
        --labels data/labels-court-kp-dense \
        --images <dir with <stem>.jpg> \
        --court /home/andrii/Projects/vsoccer/court_reg_refs/court_keypoints_29_ipbl.json \
        --out data/ipbl-court-pose [--val-frac 0.18]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_GAME = re.compile(r"^(.*?)_\d{3}_\d{5}$")
FLIP_IDX = list(range(5)) + list(range(17, 29)) + list(range(5, 17))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--court", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.18)
    ap.add_argument("--flat", action="store_true",
                    help="no train/val split: flat images/ + labels/ (for "
                         "cross-validation setups; group folds BY GAME, the "
                         "stem prefix, to avoid near-duplicate leakage)")
    a = ap.parse_args()

    court = json.loads(Path(a.court).read_text())
    id_order = sorted(court, key=int)
    name2id = {court[k]["name"]: int(k) for k in id_order}
    n_kpt = len(id_order)

    by_game: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(Path(a.labels).glob("*.json")):
        m = _GAME.match(f.stem)
        by_game[m.group(1) if m else f.stem].append(f)
    total = sum(len(v) for v in by_game.values())
    val_games, acc = set(), 0
    for g in sorted(by_game, key=lambda g: len(by_game[g])):  # smallest games -> val first
        if acc / total >= a.val_frac:
            break
        val_games.add(g)
        acc += len(by_game[g])

    out = Path(a.out)
    splits = ("",) if a.flat else ("train", "val")
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "": 0}
    missing_img = 0
    for game, files in by_game.items():
        split = "" if a.flat else ("val" if game in val_games else "train")
        for f in files:
            img = Path(a.images) / f"{f.stem}.jpg"
            if not img.exists():
                missing_img += 1
                continue
            doc = json.loads(f.read_text())
            det = doc["detections"][0]
            w, h = doc["width"], doc["height"]
            b = det["bbox"]
            cx = (b["x1"] + b["x2"]) / 2 / w
            cy = (b["y1"] + b["y2"]) / 2 / h
            bw = (b["x2"] - b["x1"]) / w
            bh = (b["y2"] - b["y1"]) / h
            kp = {name2id[k["name"]]: k for k in det.get("keypoints", []) if k["name"] in name2id}
            parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
            for kid in range(n_kpt):
                p = kp.get(kid)
                if p is None:
                    parts.append("0 0 0")
                else:
                    v = 1 if p.get("score") == 0.5 else 2
                    parts.append(f"{p['x'] / w:.6f} {p['y'] / h:.6f} {v}")
            (out / "labels" / split / f"{f.stem}.txt").write_text(" ".join(parts) + "\n")
            shutil.copyfile(img, out / "images" / split / img.name)
            counts[split] += 1

    if a.flat:
        yaml = (
            f"# FLAT export (no split) — build per-fold yamls yourself; group "
            f"folds BY GAME (stem prefix) to avoid near-duplicate leakage.\n"
            f"path: .\ntrain: images\nval: images\n"
            f"kpt_shape: [{n_kpt}, 3]\n"
            f"flip_idx: {FLIP_IDX}\n"
            f"names:\n  0: court\n"
        )
    else:
        yaml = (
            f"path: .\ntrain: images/train\nval: images/val\n"
            f"kpt_shape: [{n_kpt}, 3]\n"
            f"flip_idx: {FLIP_IDX}\n"
            f"names:\n  0: court\n"
        )
    (out / "dataset.yaml").write_text(yaml)
    if a.flat:
        print(f"exported flat={counts['']} | missing images: {missing_img}")
    else:
        print(f"exported train={counts['train']} val={counts['val']} "
              f"(val games: {sorted(val_games)}) | missing images: {missing_img}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
