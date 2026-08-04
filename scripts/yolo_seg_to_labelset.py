#!/usr/bin/env python3
"""THROWAWAY — YOLO-seg labels -> neutral-schema label set (masks as COCO RLE).

Bridges a YOLO segmentation export back into the framework so it can reach a
Label Studio brush project via `import-ls-cloud`. Polygons are filled to RLE
via geometry.polygon_to_rle; category names come from --names (index order).

    uv run python scripts/yolo_seg_to_labelset.py \
        --labels-dir <dir with <stem>.txt> \
        --image-prefix s3://bucket/datasets/<d>/frames/<g> \
        --out-dir <local dir for <stem>.json> \
        --names ball,rim,player,referee,scoreboard \
        --width 1280 --height 720 --source yolo26x_1280-pseudo

Assumes uniform image dimensions (verify a sample first). Upload the out-dir
to datasets/<d>/labels-<name>/<g>/ afterwards.
"""

import argparse
from pathlib import Path

from labeling_t.geometry import polygon_to_rle
from labeling_t.schema import BBox, Detection, ImageLabels


def convert_file(txt: Path, names: list[str], w: int, h: int,
                 image_prefix: str, source: str) -> ImageLabels:
    dets: list[Detection] = []
    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least 3 points
            continue
        cls = int(parts[0])
        vals = [float(v) for v in parts[1:]]
        pts = [(min(max(vals[i] * w, 0.0), float(w)),
                min(max(vals[i + 1] * h, 0.0), float(h)))
               for i in range(0, len(vals) - 1, 2)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        dets.append(Detection(
            bbox=BBox(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)),
            category=names[cls],
            source=source,
            mask=polygon_to_rle(pts, w, h),
        ))
    return ImageLabels(image_path=f"{image_prefix}/{txt.stem}.jpg",
                       width=w, height=h, detections=dets)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--image-prefix", required=True,
                    help="frames prefix the image_path should point at")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--names", required=True,
                    help="comma-separated class names in YOLO index order")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--source", default="yolo-seg")
    a = ap.parse_args()

    names = [n.strip() for n in a.names.split(",")]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    txts = sorted(Path(a.labels_dir).glob("*.txt"))
    n_det = 0
    for i, txt in enumerate(txts):
        labels = convert_file(txt, names, a.width, a.height,
                              a.image_prefix.rstrip("/"), a.source)
        n_det += len(labels.detections)
        (out / f"{txt.stem}.json").write_text(labels.model_dump_json())
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(txts)}")
    print(f"done: {len(txts)} files, {n_det} detections -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
