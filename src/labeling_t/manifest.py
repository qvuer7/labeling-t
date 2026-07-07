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
from datetime import datetime, timezone

from .layout import DatasetLayout
from .storage import Storage, open_storage

_IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _path(layout: DatasetLayout) -> str:
    return f"{layout.root}/manifest.json"


def _scan(storage: Storage, root: str) -> dict[str, collections.Counter]:
    """ONE listing of the dataset root -> per-set per-group counts:
    {"frames": Counter, "labels": Counter, "labels-<name>": Counter, ...}.
    Every labels[-<name>]/verified[-<name>] leaf is counted — named sets used
    to be invisible (the old code scanned only the three standard prefixes and
    the manifest silently went stale; REVIEW.md finding). Only .json files
    count as labels, so failure sidecars (*.jsonl) stay invisible."""
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n = len(root) + 1
    for uri in storage.list(root + "/"):
        parts = uri[n:].split("/")
        if len(parts) < 3:  # <set>/<group>/<stem>.ext; skips manifest.json etc.
            continue
        leaf, group = parts[0], parts[1]
        if leaf == "frames" and uri.lower().endswith(_IMG):
            counts["frames"][group] += 1
        elif (leaf == "labels" or leaf.startswith("labels-")
              or leaf == "verified" or leaf.startswith("verified-")) and uri.endswith(".json"):
            counts[leaf][group] += 1
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

    counts = _scan(storage, layout.root)
    frames, labels, verified = counts["frames"], counts["labels"], counts["verified"]
    # legacy view (frames + the two DEFAULT sets), unchanged shape for old readers
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
    # full view: EVERY set (incl. named namespaces), per group and in total
    namespaces = {leaf: dict(sorted(c.items())) for leaf, c in sorted(counts.items()) if c}
    namespace_totals = {leaf: sum(c.values()) for leaf, c in namespaces.items()}

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
        "namespaces": namespaces,
        "namespace_totals": namespace_totals,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    storage.write_text(_path(layout), json.dumps(manifest, indent=2))
    return manifest
