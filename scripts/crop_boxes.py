#!/usr/bin/env python3
"""Crop home / away / timer digit boxes out of every frame in a directory.

Edit the CONFIG block below, then run:  python scripts/crop_boxes.py
Boxes are (x1, y1, x2, y2) in absolute pixels of the full frame.
"""
from pathlib import Path

from PIL import Image, ImageDraw

# ============================ CONFIG — edit these =============================
DIR = "/home/andrii/Projects/Labeling-T/data/match_samples/vladivostok_vs_kaluga_02"   # directory of images to process

 
HOME  = (230, 620, 270, 640)
AWAY  = (280, 620, 310, 640)
TIMER = (245, 645, 300, 665)

OUT = "data/crops"
SAVE_OVERLAY = True            # also save each frame with the boxes drawn, to check placement
# =============================================================================

BOXES = {"home": HOME, "away": AWAY, "timer": TIMER}
COLORS = {"home": "#22c55e", "away": "#f97316", "timer": "#3b82f6"}
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

out = Path(OUT)
out.mkdir(parents=True, exist_ok=True)
images = sorted(f for f in Path(DIR).iterdir() if f.suffix.lower() in _EXTS)
print(f"{len(images)} images in {DIR}\n")

for img_path in images:
    img = Image.open(img_path).convert("RGB")
    stem = img_path.stem
    for name, box in BOXES.items():
        img.crop(box).save(out / f"{stem}_{name}.jpg", quality=95)
    if SAVE_OVERLAY:
        ov = img.copy()
        d = ImageDraw.Draw(ov)
        for name, box in BOXES.items():
            d.rectangle(box, outline=COLORS[name], width=2)
        ov.save(out / f"{stem}_overlay.jpg", quality=90)
    print(f"  {stem}  -> home/away/timer" + (" + overlay" if SAVE_OVERLAY else ""))

print(f"\ndone: {len(images)} images x {len(BOXES)} boxes -> {out}/")
