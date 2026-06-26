"""labeling-t CLI — plain args, no job.yaml (that's a post-second-dataset add).

Subcommands map to the pipeline stages:

    prelabel        images  ──► per-frame label JSON   (calls remote vLLM)
    import-ls       labels  ──► Label Studio project    (needs LS server)
    from-ls         LS export ──► per-frame label JSON
    to-coco         labels  ──► COCO annotations json
    ls-config       categories ──► print labeling-config XML

`labels` on disk = a directory of <stem>.json files, each one ImageLabels.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from .config import load_env
from .schema import ImageLabels

_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _find_images(directory: str) -> list[str]:
    out: list[str] = []
    for pat in _IMAGE_GLOBS:
        out.extend(glob.glob(str(Path(directory) / pat)))
    return sorted(out)


def _load_labels(labels_dir: str) -> list[ImageLabels]:
    files = sorted(glob.glob(str(Path(labels_dir) / "*.json")))
    return [
        ImageLabels.model_validate_json(Path(f).read_text())
        for f in files
        if Path(f).name != "failures.jsonl"
    ]


def _cmd_prelabel(a: argparse.Namespace) -> int:
    from .model_client import VLLMClient
    from .models import get_spec
    from .prelabel import prelabel

    try:
        spec = get_spec(a.model)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1
    images = _find_images(a.images)
    if not images:
        print(f"no images found in {a.images}", file=sys.stderr)
        return 1
    cmap = json.loads(Path(a.category_map).read_text()) if a.category_map else None
    try:
        client = VLLMClient.from_env(spec, categories=a.categories or None)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    with client:
        out = prelabel(
            images, client, a.out,
            category_map=cmap, min_score=a.min_score,
            strict_categories=a.strict_categories, max_concurrency=a.concurrency,
        )
    print(f"labeled {len(out)}/{len(images)} images -> {a.out}")
    return 0


def _cmd_import_ls(a: argparse.Namespace) -> int:  # pragma: no cover - needs LS server
    from .adapters.label_studio import import_to_label_studio

    labels = _load_labels(a.labels)
    if not labels:
        print(f"no label files in {a.labels}", file=sys.stderr)
        return 1
    project = import_to_label_studio(
        labels, base_url=a.url, api_key=a.api_key,
        project_title=a.project, categories=a.categories,
        image_base_url=a.image_base_url, image_root=a.image_root,
    )
    print(f"imported {len(labels)} tasks into LS project {project.id} ({a.url})")
    return 0


def _cmd_from_ls(a: argparse.Namespace) -> int:
    from .adapters.label_studio import from_label_studio

    export = json.loads(Path(a.export).read_text())
    labels = from_label_studio(export, result_source=a.source)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for img in labels:
        (out / (Path(img.image_path).stem + ".json")).write_text(img.model_dump_json())
    print(f"wrote {len(labels)} verified label files -> {a.out}")
    return 0


def _cmd_frames(a: argparse.Namespace) -> int:
    from .frames import frames_from_videos
    from .layout import DatasetLayout

    game = a.game or a.videos.rstrip("/").split("/")[-1]
    out = DatasetLayout.from_env(a.dataset, base=a.base).frames(game)
    n = frames_from_videos(a.videos, out, stride=a.stride)
    print(f"done: {n} frames -> {out}")
    return 0


def _cmd_to_coco(a: argparse.Namespace) -> int:
    from .adapters.coco import to_coco

    labels = _load_labels(a.labels)
    classes = a.classes or None
    n_imgs, n_anns = to_coco(labels, a.out, classes=classes)
    print(f"wrote COCO: {n_imgs} images, {n_anns} annotations -> {a.out}")
    return 0


def _cmd_ls_config(a: argparse.Namespace) -> int:
    from .adapters.label_studio import generate_label_config

    sys.stdout.write(generate_label_config(a.categories))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="labeling-t")
    sub = p.add_subparsers(dest="cmd", required=True)

    pre = sub.add_parser("prelabel", help="run a registered model over an image folder")
    pre.add_argument("--images", required=True)
    pre.add_argument("--out", required=True)
    pre.add_argument("--model", default="qwen3_vl", help="model spec key (labeling_t/models.py)")
    pre.add_argument("--categories", default=None, type=_csv, help="override the spec's default categories")
    pre.add_argument("--category-map", default=None, help="JSON file: model label -> category")
    pre.add_argument("--min-score", type=float, default=0.0)
    pre.add_argument("--strict-categories", action="store_true")
    pre.add_argument("--concurrency", type=int, default=8)
    pre.set_defaults(func=_cmd_prelabel)

    fr = sub.add_parser("frames", help="extract keyframes from videos -> dataset frames (local or S3)")
    fr.add_argument("--dataset", required=True, help="dataset name (groups frames/labels/verified/export)")
    fr.add_argument("--videos", required=True, help="source video prefix, e.g. s3://ml-cv-data/streams/<game>/")
    fr.add_argument("--game", default=None, help="game/group name (default: last segment of --videos)")
    fr.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET, else 'data')")
    fr.add_argument("--stride", type=int, default=1, help="keep every Kth keyframe (default all)")
    fr.set_defaults(func=_cmd_frames)

    imp = sub.add_parser("import-ls", help="import labels + pre-annotations into Label Studio")
    imp.add_argument("--labels", required=True, help="dir of <frame>.json neutral labels")
    imp.add_argument("--url", default="http://localhost:8080")
    imp.add_argument("--api-key", required=True)
    imp.add_argument("--project", required=True)
    imp.add_argument("--categories", required=True, type=_csv)
    imp.add_argument("--image-base-url", default=None,
                     help="static server base for frames, e.g. http://localhost:8081")
    imp.add_argument("--image-root", default="data",
                     help="local dir served at --image-base-url (relpath stripped from image paths)")
    imp.set_defaults(func=_cmd_import_ls)

    frm = sub.add_parser("from-ls", help="pull verified labels from an LS export")
    frm.add_argument("--export", required=True)
    frm.add_argument("--out", required=True)
    frm.add_argument("--source", default="annotations", choices=["annotations", "predictions"])
    frm.set_defaults(func=_cmd_from_ls)

    coco = sub.add_parser("to-coco", help="export labels to COCO")
    coco.add_argument("--labels", required=True)
    coco.add_argument("--out", required=True)
    coco.add_argument("--classes", type=_csv, default=None)
    coco.set_defaults(func=_cmd_to_coco)

    cfg = sub.add_parser("ls-config", help="print a Label Studio labeling config")
    cfg.add_argument("--categories", required=True, type=_csv)
    cfg.set_defaults(func=_cmd_ls_config)

    return p


def main(argv: list[str] | None = None) -> int:
    load_env()  # populate LABELING_T_* from .env before defaults are read
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
