#!/usr/bin/env python3
"""Crop home / away / timer boxes RELATIVE to the scoreboard mask (for type1,
whose width varies per match).

For each image it finds the paired mask JSON (<stem>.json next to the image, as
downloaded into data/match_samples/), takes the 'scoreboard' detection's box as
the anchor, and places the home/away/timer boxes as FRACTIONS of that box — so
one fraction-template works across all type1 widths.

Edit the CONFIG block, then run:  python scripts/crop_relative.py
Fractions are (fx1, fy1, fx2, fy2) in 0..1 of the scoreboard box
(0,0 = its top-left, 1,1 = its bottom-right).
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

# ============================ CONFIG — edit these =============================
DIR = "data/match_samples/kirovograd_vs_mytischi_01"   # a type1 match (images + paired .json masks)
ANCHOR = "scoreboard"                                 # detection category to anchor on

# boxes as fractions of the scoreboard box — TUNE these using the overlay:
HOME_REL  = (0.46, 0.30, 0.54, 0.73)
AWAY_REL  = (0.56, 0.30, 0.64, 0.73)
TIMER_REL = (0.50, 0.74, 0.62, 0.98)

OUT = "data/crops_type1"
SAVE_OVERLAY = True     # draws the scoreboard box + the 3 sub-boxes, to tune the fractions
# =============================================================================

RELS = {"home": HOME_REL, "away": AWAY_REL, "timer": TIMER_REL}
COLORS = {"home": "#22c55e", "away": "#f97316", "timer": "#3b82f6"}
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def scoreboard_box(mask_json: Path):
    """Largest ANCHOR-category box in the label, as (x1, y1, x2, y2). None if none."""
    dets = json.loads(mask_json.read_text()).get("detections", [])
    boxes = [d["bbox"] for d in dets if d.get("category") == ANCHOR]
    if not boxes:
        return None
    b = max(boxes, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return b["x1"], b["y1"], b["x2"], b["y2"]


def rel_to_abs(rel, anchor):
    sx1, sy1, sx2, sy2 = anchor
    w, h = sx2 - sx1, sy2 - sy1
    fx1, fy1, fx2, fy2 = rel
    return (round(sx1 + fx1 * w), round(sy1 + fy1 * h),
            round(sx1 + fx2 * w), round(sy1 + fy2 * h))


out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)
images = sorted(f for f in Path(DIR).iterdir() if f.suffix.lower() in _EXTS)
print(f"{len(images)} images in {DIR}\n")

done = skipped = 0
for img_path in images:
    mask_json = img_path.with_suffix(".json")
    if not mask_json.exists():
        print(f"  {img_path.stem}: no paired mask .json — skip"); skipped += 1; continue
    anchor = scoreboard_box(mask_json)
    if anchor is None:
        print(f"  {img_path.stem}: no '{ANCHOR}' in mask — skip"); skipped += 1; continue

    img = Image.open(img_path).convert("RGB")
    boxes = {name: rel_to_abs(rel, anchor) for name, rel in RELS.items()}
    for name, box in boxes.items():
        img.crop(box).save(out / f"{img_path.stem}_{name}.jpg", quality=95)
    if SAVE_OVERLAY:
        ov = img.copy()
        d = ImageDraw.Draw(ov)
        d.rectangle([round(v) for v in anchor], outline="#ffffff", width=1)  # the anchor box
        for name, box in boxes.items():
            d.rectangle(box, outline=COLORS[name], width=2)
        ov.save(out / f"{img_path.stem}_overlay.jpg", quality=90)
    done += 1
    print(f"  {img_path.stem}: scoreboard W={anchor[2]-anchor[0]:.0f} -> home/away/timer")

print(f"\ndone: {done} cropped, {skipped} skipped -> {out}/")
