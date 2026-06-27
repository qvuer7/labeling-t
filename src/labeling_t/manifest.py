"""Dataset manifest — `manifest.json` on S3: the queryable index + metadata.

Read it instead of listing S3 over and over. It is REGENERABLE from S3 (the
per-group counts are scanned), plus declared metadata (categories, source,
extraction params, model) that isn't derivable. S3 stays the source of truth;
the manifest is a generated index, so it can never fatally drift — rebuild any
time with `build_manifest`.

    datasets/<dataset>/manifest.json
"""

from __future__ import annotations

import collections
import json

from .layout import DatasetLayout
from .storage import Storage, open_storage

_IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _path(layout: DatasetLayout) -> str:
    return f"{layout.root}/manifest.json"


def _counts_by_group(storage: Storage, stage_root: str, exts: tuple[str, ...]) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    n = len(stage_root) + 1
    for uri in storage.list(stage_root + "/"):
        if uri.lower().endswith(exts):
            counts[uri[n:].split("/")[0]] += 1  # <group>/<stem>.ext -> <group>
    return counts


def load_manifest(dataset: str, *, base: str | None = None, storage: Storage | None = None) -> dict | None:
    layout = DatasetLayout.from_env(dataset, base=base)
    storage = storage or open_storage(layout.root)
    try:
        return json.loads(storage.read_bytes(_path(layout)).decode())
    except Exception:
        return None


def build_manifest(
    dataset: str,
    *,
    base: str | None = None,
    storage: Storage | None = None,
    categories: list[str] | None = None,
    source: str | None = None,
    stride: int | None = None,
    model: str | None = None,
) -> dict:
    """Scan S3 for per-group counts, merge declared metadata (new args win, else
    keep what's already in the manifest), write and return manifest.json."""
    layout = DatasetLayout.from_env(dataset, base=base)
    storage = storage or open_storage(layout.root)
    prev = load_manifest(dataset, base=base, storage=storage) or {}

    frames = _counts_by_group(storage, layout.frames(), _IMG)
    labels = _counts_by_group(storage, layout.labels(), (".json",))
    verified = _counts_by_group(storage, layout.verified(), (".json",))
    groups = {
        g: {"frames": frames.get(g, 0), "labels": labels.get(g, 0), "verified": verified.get(g, 0)}
        for g in sorted(set(frames) | set(labels) | set(verified))
    }
    totals = {
        "frames": sum(frames.values()),
        "labels": sum(labels.values()),
        "verified": sum(verified.values()),
        "groups": len(groups),
    }

    extraction = {"method": "keyframes", "stride": stride} if stride is not None else prev.get("extraction")
    manifest = {
        "dataset": dataset,
        "task": prev.get("task", "detection"),
        "instance_type": prev.get("instance_type", "image"),
        "categories": categories if categories is not None else prev.get("categories", []),
        "source": source or prev.get("source"),
        "extraction": extraction,
        "model": model or prev.get("model"),
        "groups": groups,
        "totals": totals,
    }
    storage.write_text(_path(layout), json.dumps(manifest, indent=2))
    return manifest
