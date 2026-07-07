"""labeling-t CLI — plain args, no job.yaml (that's a post-second-dataset add).

Subcommands map to the pipeline stages:

    prelabel        images  ──► per-frame label JSON   (calls remote vLLM)
    segment-cloud   labels  ──► same labels + Detection.mask (SAM2 box prompts)
    transcribe      labels  ──► same labels + Detection.text (hosted VLM OCR)
    import-ls       labels  ──► Label Studio project    (needs LS server)
    from-ls         LS export ──► per-frame label JSON
    to-coco         labels  ──► COCO annotations json
    ls-config       categories ──► print labeling-config XML
    stats/validate  label set ──► counts / schema violations (labelset.py)
    diff            two label sets ──► stems only-in-a/only-in-b/changed

`labels` on disk = a directory of <stem>.json files, each one ImageLabels.

Every subcommand takes --json: one machine-readable envelope on stdout
(`{"ok": ..., "result"/"error": ...}`), prose to stderr. See output.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from .config import load_env
from .output import emit, fail, json_flag, note
from .schema import ImageLabels

_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _env_arg(var: str) -> dict:
    """argparse kwargs so a flag defaults to an env var (read after load_env);
    required only when the env var is absent. Keeps URLs/keys out of commands."""
    return {"default": os.environ.get(var), "required": var not in os.environ}


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


def _ls_project_url(base_url: str, project_id) -> str:
    return f"{base_url.rstrip('/')}/projects/{project_id}"


def _cmd_prelabel(a: argparse.Namespace) -> int:
    from .model_client import client_for
    from .models import get_spec
    from .prelabel import prelabel

    try:
        spec = get_spec(a.model)
    except KeyError as exc:
        return fail(a, str(exc))
    images = _find_images(a.images)
    if not images:
        return fail(a, f"no images found in {a.images}")
    cmap = json.loads(Path(a.category_map).read_text()) if a.category_map else None
    try:
        client = client_for(spec, endpoint=a.endpoint, categories=a.categories or None)
    except ValueError as exc:
        return fail(a, str(exc))
    with client:
        out = prelabel(
            images, client, a.out,
            category_map=cmap, min_score=a.min_score,
            strict_categories=a.strict_categories, max_concurrency=a.concurrency,
        )
    return emit(
        a,
        {"labeled": len(out), "total": len(images), "out": a.out,
         "failures_file": str(Path(a.out) / "failures.jsonl")},
        f"labeled {len(out)}/{len(images)} images -> {a.out}",
    )


def _cmd_import_ls(a: argparse.Namespace) -> int:  # pragma: no cover - needs LS server
    from .adapters.label_studio import import_to_label_studio

    labels = _load_labels(a.labels)
    if not labels:
        return fail(a, f"no label files in {a.labels}")
    try:
        project = import_to_label_studio(
            labels, base_url=a.url, api_key=a.api_key,
            project_title=a.project, categories=a.categories,
            image_base_url=a.image_base_url, image_root=a.image_root,
        )
    except ValueError as exc:  # e.g. project title over LS's length limit
        return fail(a, str(exc))
    url = _ls_project_url(a.url, project.id)
    return emit(
        a,
        {"tasks": len(labels), "project_id": project.id, "project_url": url},
        f"imported {len(labels)} tasks into LS project {project.id} ({url})",
    )


def _cmd_import_ls_cloud(a: argparse.Namespace) -> int:  # pragma: no cover - needs LS + S3
    from .adapters.label_studio import import_to_label_studio
    from .layout import DatasetLayout
    from .schema import ImageLabels
    from .storage import open_storage

    labels_prefix = DatasetLayout.from_env(a.dataset, base=a.base).labels(a.group, a.labels_name)
    storage = open_storage(labels_prefix)
    uris = [u for u in storage.list(labels_prefix + "/") if u.endswith(".json")]
    if not uris:
        return fail(a, f"no labels under {labels_prefix} (run prelabel-cloud first)")
    images = [ImageLabels.model_validate_json(storage.read_bytes(u).decode()) for u in uris]
    try:
        project = import_to_label_studio(
            images, base_url=a.url, api_key=a.api_key, project_title=a.project,
            categories=a.categories,
            presign=lambda uri: storage.presigned_url(uri, a.ttl),  # frame URI -> presigned URL
            control=a.mask_format if a.masks else "rectangle",
        )
    except ValueError as exc:  # e.g. project title over LS's length limit
        return fail(a, str(exc))
    url = _ls_project_url(a.url, project.id)
    return emit(
        a,
        {"tasks": len(images), "project_id": project.id, "project_url": url},
        f"imported {len(images)} tasks into LS project {project.id} ({url})",
    )


def _cmd_from_ls(a: argparse.Namespace) -> int:
    from .adapters.label_studio import from_label_studio

    export = json.loads(Path(a.export).read_text())
    labels = from_label_studio(export, result_source=a.source)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for img in labels:
        (out / (Path(img.image_path).stem + ".json")).write_text(img.model_dump_json())
    return emit(a, {"files": len(labels), "out": a.out},
                f"wrote {len(labels)} verified label files -> {a.out}")


def _refresh_manifest(dataset: str, base: str | None) -> None:
    """Best-effort: keep manifest.json current after a stage. Never fatal."""
    try:
        from .manifest import build_manifest
        build_manifest(dataset, base=base)
    except Exception:  # noqa: BLE001
        pass


def _cmd_manifest(a: argparse.Namespace) -> int:
    from .manifest import build_manifest, load_manifest

    if a.show:
        m = load_manifest(a.dataset, base=a.base)
        if m is None:
            return fail(a, "no manifest yet (run without --show to build one)")
    else:
        m = build_manifest(a.dataset, base=a.base, categories=a.categories,
                           source=a.source, stride=a.stride, model=a.model)
    if a.json:
        return emit(a, m)
    print(json.dumps(m, indent=2))
    return 0


def _cmd_from_ls_cloud(a: argparse.Namespace) -> int:  # pragma: no cover - needs LS + S3
    from .layout import DatasetLayout
    from .verify import pull_verified

    n = pull_verified(
        a.dataset, a.group, url=a.url, api_key=a.api_key,
        project_id=a.project_id, base=a.base, name=a.name,
    )
    verified_prefix = DatasetLayout.from_env(a.dataset, base=a.base).verified(a.group, a.name)
    rc = emit(a, {"pulled": n, "prefix": verified_prefix},
              f"pulled {n} verified labels -> {verified_prefix}")
    _refresh_manifest(a.dataset, a.base)
    return rc


def _cmd_frames(a: argparse.Namespace) -> int:
    from .frames import VIDEO_EXTS, frames_from_videos
    from .layout import DatasetLayout
    from .storage import open_storage

    layout = DatasetLayout.from_env(a.dataset, base=a.base)
    if a.all_groups:
        root = a.videos.rstrip("/")
        storage = open_storage(a.videos)
        keys = storage.list(root + "/")
        groups = sorted({k[len(root) + 1:].split("/")[0] for k in keys if k.lower().endswith(VIDEO_EXTS)})
        total = 0
        for i, g in enumerate(groups, 1):
            note(a, f"[{i}/{len(groups)}] {g}")
            total += frames_from_videos(f"{root}/{g}/", layout.frames(g), stride=a.stride)
        rc = emit(a, {"frames": total, "groups": len(groups), "out": layout.frames("")},
                  f"done: {total} frames across {len(groups)} groups -> {layout.frames('')}/")
        _refresh_manifest(a.dataset, a.base)
        return rc

    group = a.group or a.videos.rstrip("/").split("/")[-1]
    out = layout.frames(group)
    n = frames_from_videos(a.videos, out, stride=a.stride)
    rc = emit(a, {"frames": n, "out": out}, f"done: {n} frames -> {out}")
    _refresh_manifest(a.dataset, a.base)
    return rc


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _cmd_ingest_images(a: argparse.Namespace) -> int:
    from .ingest import ingest_images
    from .layout import DatasetLayout
    from .storage import open_storage

    dest = DatasetLayout.from_env(a.dataset, base=a.base).frames(a.group)
    storage = open_storage(dest)
    try:
        uploaded, total = ingest_images(a.src, dest, storage=storage, max_concurrency=a.concurrency)
    except (NotADirectoryError, FileNotFoundError) as exc:
        return fail(a, str(exc))
    if total == 0:
        return fail(a, f"no images found in {a.src}")
    skipped = total - uploaded
    note_ = f" ({skipped} already present)" if skipped else ""
    rc = emit(a, {"uploaded": uploaded, "total": total, "skipped": skipped, "dest": dest},
              f"uploaded {uploaded}/{total} images -> {dest}{note_}")
    _refresh_manifest(a.dataset, a.base)
    return rc


def _cmd_prelabel_cloud(a: argparse.Namespace) -> int:  # pragma: no cover - needs vLLM + S3
    from .layout import DatasetLayout
    from .model_client import client_for
    from .models import get_spec
    from .prelabel import prelabel_cloud
    from .storage import open_storage

    try:
        spec = get_spec(a.model)
    except KeyError as exc:
        return fail(a, str(exc))
    layout = DatasetLayout.from_env(a.dataset, base=a.base)
    frames_prefix, labels_prefix = layout.frames(a.group), layout.labels(a.group, a.labels_name)
    storage = open_storage(frames_prefix)
    frames = [u for u in storage.list(frames_prefix + "/") if u.lower().endswith(_IMAGE_SUFFIXES)]
    if not frames:
        return fail(a, f"no frames under {frames_prefix}")
    cmap = json.loads(Path(a.category_map).read_text()) if a.category_map else None
    try:
        client = client_for(spec, endpoint=a.endpoint, categories=a.categories or None)
    except ValueError as exc:
        return fail(a, str(exc))
    with client:
        n = prelabel_cloud(
            frames, client, labels_prefix, storage=storage, category_map=cmap,
            min_score=a.min_score, strict_categories=a.strict_categories,
            max_concurrency=a.concurrency,
        )
    rc = emit(a, {"labeled": n, "total": len(frames), "prefix": labels_prefix,
                  "failures_file": f"{labels_prefix}/failures.jsonl"},
              f"labeled {n}/{len(frames)} frames -> {labels_prefix}")
    _refresh_manifest(a.dataset, a.base)
    return rc


def _transcribe_client(a: argparse.Namespace):
    """Resolve + validate the OCR spec and build its chat client. Raises
    ValueError on a bad spec/config. Shared by transcribe / transcribe-cloud."""
    from dataclasses import replace

    from .model_client import client_for
    from .models import get_spec

    try:
        spec = get_spec(a.model)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    if spec.backend == "transformers":
        raise ValueError(f"model {spec.key!r} runs on the transformers backend; "
                         "transcribe needs a chat backend (openai/gemini/vllm)")
    if a.prompt:
        spec = replace(spec, prompt=a.prompt)  # frozen dataclass -> copy
    # a transcription is a few tokens; cap generation accordingly.
    return client_for(spec, endpoint=a.endpoint, max_tokens=64)


def _cmd_transcribe(a: argparse.Namespace) -> int:
    from .transcribe import FAILURES_NAME, transcribe

    try:
        client = _transcribe_client(a)
    except ValueError as exc:
        return fail(a, str(exc))
    with client:
        n = transcribe(
            a.labels, client, categories=a.categories, pad=a.pad,
            images_dir=a.images, max_concurrency=a.concurrency,
        )
    return emit(a, {"transcribed": n, "labels": a.labels,
                    "failures_file": str(Path(a.labels) / FAILURES_NAME)},
                f"transcribed {n} label files in place -> {a.labels}")


def _cmd_transcribe_cloud(a: argparse.Namespace) -> int:  # pragma: no cover - needs vendor API + S3
    from .layout import DatasetLayout
    from .storage import open_storage
    from .transcribe import FAILURES_NAME, transcribe_cloud

    try:
        client = _transcribe_client(a)
    except ValueError as exc:
        return fail(a, str(exc))
    layout = DatasetLayout.from_env(a.dataset, base=a.base)
    labels_prefix = layout.labels(a.group, a.labels_name)
    to_prefix = layout.labels(a.group, a.to_name) if a.to_name else None
    storage = open_storage(labels_prefix)
    with client:
        n = transcribe_cloud(
            labels_prefix, client, storage=storage, categories=a.categories,
            pad=a.pad, to_prefix=to_prefix, max_concurrency=a.concurrency,
        )
    out = to_prefix or labels_prefix
    rc = emit(a, {"transcribed": n, "prefix": out, "failures_file": f"{out}/{FAILURES_NAME}"},
              f"transcribed {n} label files -> {out}")
    _refresh_manifest(a.dataset, a.base)
    return rc


def _cmd_segment_cloud(a: argparse.Namespace) -> int:  # pragma: no cover - needs model-server + S3
    from .layout import DatasetLayout
    from .model_client import client_for
    from .models import get_spec
    from .segment import FAILURES_NAME, segment_cloud
    from .storage import open_storage

    try:
        spec = get_spec(a.model)
    except KeyError as exc:
        return fail(a, str(exc))
    if spec.backend != "transformers":
        return fail(a, f"model {spec.key!r} is not a segmenter on our model-server "
                       "(transformers backend required, e.g. sam2)")
    layout = DatasetLayout.from_env(a.dataset, base=a.base)
    labels_prefix = layout.labels(a.group, a.labels_name)
    to_prefix = layout.labels(a.group, a.to_name) if a.to_name else None
    storage = open_storage(labels_prefix)
    try:
        client = client_for(spec, endpoint=a.endpoint)
    except ValueError as exc:
        return fail(a, str(exc))
    with client:
        n = segment_cloud(
            labels_prefix, client, storage=storage, categories=a.categories,
            to_prefix=to_prefix, max_concurrency=a.concurrency,
        )
    out = to_prefix or labels_prefix
    rc = emit(a, {"segmented": n, "prefix": out, "failures_file": f"{out}/{FAILURES_NAME}"},
              f"segmented {n} label files -> {out}")
    _refresh_manifest(a.dataset, a.base)
    return rc


def _resolve_set(a: argparse.Namespace, *, dir_attr: str = "labels",
                 sel_attr: str = "set") -> tuple[str, str, object]:
    """(display name, prefix, storage) for a set named either by a local dir
    (--labels DIR) or by dataset coordinates (--dataset/--group/--set SELECTOR).
    Raises ValueError with the agent-actionable message."""
    from .layout import DatasetLayout
    from .storage import LocalStorage, open_storage

    directory = getattr(a, dir_attr, None)
    selector = getattr(a, sel_attr, None)
    if directory and (a.dataset or selector):
        raise ValueError(f"--{dir_attr} is a local-dir mode; don't combine it with --dataset/--{sel_attr}")
    if directory:
        prefix = directory.rstrip("/")
        return directory, prefix, LocalStorage()
    if not (a.dataset and a.group and selector):
        raise ValueError(f"name a set: either --{dir_attr} DIR, or --dataset D --group G --{sel_attr} "
                         "SELECTOR (labels | labels-<name> | verified | verified-<name>)")
    prefix = DatasetLayout.from_env(a.dataset, base=a.base).set_prefix(a.group, selector)
    return selector, prefix, open_storage(prefix)


def _cmd_stats(a: argparse.Namespace) -> int:
    from .labelset import set_stats

    try:
        name, prefix, storage = _resolve_set(a)
    except ValueError as exc:
        return fail(a, str(exc))
    res = {"set": name, **set_stats(prefix, storage=storage)}
    if a.json:
        return emit(a, res, f"{res['files']} files, {res['detections']} detections")
    print(json.dumps(res, indent=2))
    return 0


def _cmd_validate(a: argparse.Namespace) -> int:
    from .labelset import set_validate

    try:
        name, prefix, storage = _resolve_set(a)
    except ValueError as exc:
        return fail(a, str(exc))
    res = set_validate(prefix, storage=storage)
    if res["violations"]:
        return fail(a, f"{len(res['violations'])}/{res['files']} files violate the schema "
                       f"(showing up to {a.limit})",
                    result={"set": name, "files": res["files"], "valid": res["valid"],
                            "violations": res["violations"][:a.limit],
                            "violations_total": len(res["violations"])})
    return emit(a, {"set": name, "files": res["files"], "valid": res["valid"],
                    "violations": [], "violations_total": 0},
                f"OK: {res['files']} files, all schema-valid ({name})")


def _cmd_diff(a: argparse.Namespace) -> int:
    from .labelset import set_diff

    try:
        name_a, prefix_a, storage_a = _resolve_set(a, dir_attr="a_dir", sel_attr="a")
        name_b, prefix_b, storage_b = _resolve_set(a, dir_attr="b_dir", sel_attr="b")
    except ValueError as exc:
        return fail(a, str(exc))
    res = set_diff(prefix_a, prefix_b, storage_a=storage_a, storage_b=storage_b)
    res = {"a": name_a, "b": name_b, **res}
    msg = (f"a={name_a} b={name_b}: only-in-a {len(res['only_in_a'])}, "
           f"only-in-b {len(res['only_in_b'])}, changed {len(res['changed'])}, "
           f"identical {res['identical']} ({res['byte_identical']} byte-identical)")
    if a.json:
        return emit(a, res, msg)
    print(json.dumps(res, indent=2))
    print(msg)
    return 0


def _cmd_to_coco(a: argparse.Namespace) -> int:
    from .adapters.coco import to_coco

    labels = _load_labels(a.labels)
    classes = a.classes or None
    n_imgs, n_anns = to_coco(labels, a.out, classes=classes)
    return emit(a, {"images": n_imgs, "annotations": n_anns, "out": a.out},
                f"wrote COCO: {n_imgs} images, {n_anns} annotations -> {a.out}")


def _cmd_ls_config(a: argparse.Namespace) -> int:
    from .adapters.label_studio import generate_label_config

    cfg = generate_label_config(a.categories)
    if a.json:
        return emit(a, {"xml": cfg})
    sys.stdout.write(cfg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="labeling-t")
    sub = p.add_subparsers(dest="cmd", required=True)
    jf = [json_flag()]  # every subcommand takes --json (envelope on stdout)

    pre = sub.add_parser("prelabel", help="run a registered model over an image folder", parents=jf)
    pre.add_argument("--images", required=True)
    pre.add_argument("--out", required=True)
    pre.add_argument("--model", default="qwen3_vl", help="model spec key (labeling_t/models.py)")
    pre.add_argument("--endpoint", default=None,
                     help="model endpoint URL (default: newest recorded pod for the model, "
                          "see `labeling-t-runpod status`)")
    pre.add_argument("--categories", default=None, type=_csv, help="override the spec's default categories")
    pre.add_argument("--category-map", default=None, help="JSON file: model label -> category")
    pre.add_argument("--min-score", type=float, default=0.0)
    pre.add_argument("--strict-categories", action="store_true")
    pre.add_argument("--concurrency", type=int, default=8)
    pre.set_defaults(func=_cmd_prelabel)

    fr = sub.add_parser("frames", help="extract keyframes from videos -> dataset frames (local or S3)", parents=jf)
    fr.add_argument("--dataset", required=True, help="dataset name (groups frames/labels/verified/export)")
    fr.add_argument("--videos", required=True, help="source video prefix, e.g. s3://ml-cv-data/streams/<game>/")
    fr.add_argument("--group", default=None, help="group/partition name (default: last segment of --videos)")
    fr.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET, else 'data')")
    fr.add_argument("--stride", type=int, default=1, help="keep every Kth keyframe (default all)")
    fr.add_argument("--all-groups", action="store_true",
                    help="treat --videos as the source root and process every subfolder as a group")
    fr.set_defaults(func=_cmd_frames)

    ig = sub.add_parser("ingest-images", help="upload a local image folder into a dataset group's frames (storage)",
                        parents=jf)
    ig.add_argument("--src", required=True, help="local directory of images")
    ig.add_argument("--dataset", required=True)
    ig.add_argument("--group", required=True, help="group/partition name, e.g. all")
    ig.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET, else 'data')")
    ig.add_argument("--concurrency", type=int, default=8)
    ig.set_defaults(func=_cmd_ingest_images)

    pc = sub.add_parser("prelabel-cloud", help="label a dataset's frames in S3 (presigned URL -> vLLM -> labels in S3)",
                        parents=jf)
    pc.add_argument("--dataset", required=True)
    pc.add_argument("--group", required=True)
    pc.add_argument("--model", default="qwen3_vl", help="model spec key")
    pc.add_argument("--endpoint", default=None,
                    help="model endpoint URL (default: newest recorded pod for the model)")
    pc.add_argument("--labels-name", default="", help="namespace pre-labels into labels-<name>/ "
                    "(keeps several models' pre-labels apart; default writes to labels/)")
    pc.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET)")
    pc.add_argument("--categories", default=None, type=_csv, help="override spec default categories")
    pc.add_argument("--category-map", default=None, help="JSON file: model label -> category")
    pc.add_argument("--min-score", type=float, default=0.0)
    pc.add_argument("--strict-categories", action="store_true")
    pc.add_argument("--concurrency", type=int, default=8)
    pc.set_defaults(func=_cmd_prelabel_cloud)

    sg = sub.add_parser("segment-cloud", help="fill Detection.mask: send a label set's boxes to the segmenter (SAM2)",
                        parents=jf)
    sg.add_argument("--dataset", required=True)
    sg.add_argument("--group", required=True)
    sg.add_argument("--model", default="sam2", help="segmenter spec key (transformers backend)")
    sg.add_argument("--endpoint", default=None,
                    help="model endpoint URL (default: newest recorded pod for the model)")
    sg.add_argument("--labels-name", default="", help="read labels from labels-<name>/ (default labels/)")
    sg.add_argument("--to-name", default=None,
                    help="write enriched copies to labels-<name>/ instead of rewriting the source in place")
    sg.add_argument("--categories", default=None, type=_csv,
                    help="only segment these detection categories (default: all boxes)")
    sg.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET)")
    sg.add_argument("--concurrency", type=int, default=1,
                    help="keep 1 for the transformers backend (one GPU, not reentrant)")
    sg.set_defaults(func=_cmd_segment_cloud)

    tr = sub.add_parser("transcribe", help="OCR: fill Detection.text on matching regions via a hosted VLM",
                        parents=jf)
    tr.add_argument("--labels", required=True, help="dir of <frame>.json neutral labels (rewritten in place)")
    tr.add_argument("--categories", required=True, type=_csv,
                    help="which detection categories to transcribe (the region filter)")
    tr.add_argument("--model", default="openai_ocr", help="OCR model spec key (openai_ocr / gemini_ocr)")
    tr.add_argument("--endpoint", default=None,
                    help="model endpoint URL (default: the spec's provider URL / newest recorded pod)")
    tr.add_argument("--prompt", default=None,
                    help="override the OCR prompt (literal { or } must be doubled: {{ }})")
    tr.add_argument("--pad", type=int, default=2, help="pixels of context around each crop")
    tr.add_argument("--images", default=None,
                    help="frame directory override when labels' image_path doesn't resolve (e.g. from-ls output)")
    tr.add_argument("--concurrency", type=int, default=4)
    tr.set_defaults(func=_cmd_transcribe)

    tc = sub.add_parser("transcribe-cloud", help="OCR a dataset's S3 labels: crop regions locally, text via hosted VLM",
                        parents=jf)
    tc.add_argument("--dataset", required=True)
    tc.add_argument("--group", required=True)
    tc.add_argument("--categories", required=True, type=_csv,
                    help="which detection categories to transcribe (the region filter)")
    tc.add_argument("--labels-name", default="", help="read labels from labels-<name>/ (default labels/)")
    tc.add_argument("--to-name", default=None,
                    help="write enriched copies to labels-<name>/ instead of rewriting the source in place")
    tc.add_argument("--model", default="openai_ocr", help="OCR model spec key (openai_ocr / gemini_ocr)")
    tc.add_argument("--endpoint", default=None,
                    help="model endpoint URL (default: the spec's provider URL / newest recorded pod)")
    tc.add_argument("--prompt", default=None,
                    help="override the OCR prompt (literal { or } must be doubled: {{ }})")
    tc.add_argument("--pad", type=int, default=2, help="pixels of context around each crop")
    tc.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET)")
    tc.add_argument("--concurrency", type=int, default=4)
    tc.set_defaults(func=_cmd_transcribe_cloud)

    ic = sub.add_parser("import-ls-cloud", help="import a dataset's S3 labels into LS (frames via presigned URLs)",
                        parents=jf)
    ic.add_argument("--dataset", required=True)
    ic.add_argument("--group", required=True)
    ic.add_argument("--labels-name", default="", help="read pre-labels from labels-<name>/ "
                    "(match the prelabel-cloud --labels-name; default reads labels/)")
    ic.add_argument("--url", help="hosted Label Studio base URL (default $LS_URL)", **_env_arg("LS_URL"))
    ic.add_argument("--api-key", help="LS API token (default $LS_API_KEY)", **_env_arg("LS_API_KEY"))
    ic.add_argument("--project", required=True)
    ic.add_argument("--categories", required=True, type=_csv)
    ic.add_argument("--masks", action="store_true",
                    help="verify SAM2 masks instead of boxes (see --mask-format)")
    ic.add_argument("--mask-format", choices=["polygon", "brush"], default="polygon",
                    help="with --masks: polygon (editable, default) or brush (raster mask)")
    ic.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET)")
    ic.add_argument("--ttl", type=int, default=604800, help="presigned URL lifetime seconds (default 7d)")
    ic.set_defaults(func=_cmd_import_ls_cloud)

    fc = sub.add_parser("from-ls-cloud", help="pull verified annotations from LS API -> S3 verified/", parents=jf)
    fc.add_argument("--dataset", required=True)
    fc.add_argument("--group", required=True)
    fc.add_argument("--url", help="hosted Label Studio base URL (default $LS_URL)", **_env_arg("LS_URL"))
    fc.add_argument("--api-key", help="LS API token (default $LS_API_KEY)", **_env_arg("LS_API_KEY"))
    fc.add_argument("--project-id", required=True, help="LS project id (from import-ls-cloud output)")
    fc.add_argument("--name", default="", help="namespace the verified output into verified-<name>/ "
                    "(e.g. masks), to not overwrite the box-verified verified/")
    fc.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET)")
    fc.set_defaults(func=_cmd_from_ls_cloud)

    mf = sub.add_parser("manifest", help="build/show a dataset's manifest.json (metadata + per-group counts)",
                        parents=jf)
    mf.add_argument("--dataset", required=True)
    mf.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET)")
    mf.add_argument("--show", action="store_true", help="read the stored manifest without rescanning S3")
    mf.add_argument("--categories", type=_csv, default=None, help="set the dataset's category set")
    mf.add_argument("--source", default=None, help="raw source prefix, e.g. s3://ml-cv-data/streams/")
    mf.add_argument("--stride", type=int, default=None, help="record extraction stride")
    mf.add_argument("--model", default=None, help="record the prelabel model")
    mf.set_defaults(func=_cmd_manifest)

    imp = sub.add_parser("import-ls", help="import labels + pre-annotations into Label Studio", parents=jf)
    imp.add_argument("--labels", required=True, help="dir of <frame>.json neutral labels")
    imp.add_argument("--url", default=os.environ.get("LS_URL", "http://localhost:8080"),
                     help="LS base URL (default $LS_URL or http://localhost:8080)")
    imp.add_argument("--api-key", help="LS API token (default $LS_API_KEY)", **_env_arg("LS_API_KEY"))
    imp.add_argument("--project", required=True)
    imp.add_argument("--categories", required=True, type=_csv)
    imp.add_argument("--image-base-url", default=None,
                     help="static server base for frames, e.g. http://localhost:8081")
    imp.add_argument("--image-root", default="data",
                     help="local dir served at --image-base-url (relpath stripped from image paths)")
    imp.set_defaults(func=_cmd_import_ls)

    frm = sub.add_parser("from-ls", help="pull verified labels from an LS export", parents=jf)
    frm.add_argument("--export", required=True)
    frm.add_argument("--out", required=True)
    frm.add_argument("--source", default="annotations", choices=["annotations", "predictions"])
    frm.set_defaults(func=_cmd_from_ls)

    def _set_args(sp):
        sp.add_argument("--dataset", default=None)
        sp.add_argument("--group", default=None)
        sp.add_argument("--set", default=None,
                        help="set selector: labels | labels-<name> | verified | verified-<name>")
        sp.add_argument("--labels", default=None, help="local dir of <stem>.json labels instead of --dataset/--set")
        sp.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET)")

    st = sub.add_parser("stats", help="aggregate counts for a label set "
                        "(files, detections, categories, mask/text coverage, sources)", parents=jf)
    _set_args(st)
    st.set_defaults(func=_cmd_stats)

    vd = sub.add_parser("validate", help="schema-validate every file in a label set (rc 1 on violations)",
                        parents=jf)
    _set_args(vd)
    vd.add_argument("--limit", type=int, default=20, help="max violations to report (total is always counted)")
    vd.set_defaults(func=_cmd_validate)

    df = sub.add_parser("diff", help="compare two label sets by stem (only-in-a/only-in-b/changed/identical)",
                        parents=jf)
    df.add_argument("--dataset", default=None)
    df.add_argument("--group", default=None)
    df.add_argument("--a", default=None, help="set selector for side A (labels-<name>, verified, ...)")
    df.add_argument("--b", default=None, help="set selector for side B")
    df.add_argument("--a-dir", default=None, help="local dir for side A instead of --a")
    df.add_argument("--b-dir", default=None, help="local dir for side B instead of --b")
    df.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET)")
    df.set_defaults(func=_cmd_diff)

    coco = sub.add_parser("to-coco", help="export labels to COCO", parents=jf)
    coco.add_argument("--labels", required=True)
    coco.add_argument("--out", required=True)
    coco.add_argument("--classes", type=_csv, default=None)
    coco.set_defaults(func=_cmd_to_coco)

    cfg = sub.add_parser("ls-config", help="print a Label Studio labeling config", parents=jf)
    cfg.add_argument("--categories", required=True, type=_csv)
    cfg.set_defaults(func=_cmd_ls_config)

    return p


def main(argv: list[str] | None = None) -> int:
    load_env()  # populate LABELING_T_* from .env before defaults are read
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
