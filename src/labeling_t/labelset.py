"""Label-set primitives — stats / validate / diff over any stored label set.

The dataset-state questions agents re-derive by hand every session ("how many
detections, what's masked, did the rewrite change anything?") as first-class,
machine-readable commands. Pure over the Storage protocol, so a local dir and
an S3 prefix answer identically.

Semantics worth pinning:

- stats/validate are QUERIES, not assertions: an empty prefix is ok with
  files=0, an unreadable file is counted (and reported), never fatal.
- schema_version is counted from the RAW JSON — pydantic's default would make
  a pre-versioning file (no field on disk) indistinguishable from "1".
- text follows the transcribe resume contract: None = never attempted,
  "" = attempted but illegible.
- diff joins by stem. `changed` compares ORDER-NORMALIZED detections and
  ignores image_path/width/height — a verified pull-back rewrites image_path,
  and byte-diffing would flag every file as changed. `byte_identical` is the
  sha256 count for the opposite need: never delete a source set without proof
  the copy is bit-for-bit.
"""

from __future__ import annotations

import hashlib
import json

from .schema import ImageLabels
from .storage import Storage


def _label_files(storage: Storage, prefix: str) -> dict[str, str]:
    """{stem: uri} for the label files under `prefix`."""
    return {
        u.rsplit("/", 1)[-1][:-5]: u
        for u in storage.list(prefix.rstrip("/") + "/")
        if u.endswith(".json")
    }


def label_stems(prefix: str, *, storage: Storage) -> set[str]:
    """The file stems of a label set — the join key across stages (frames,
    labels, verified share stems). Feeds --frames-from subsetting."""
    return set(_label_files(storage, prefix))


def set_stats(prefix: str, *, storage: Storage) -> dict:
    """Aggregate counts for one label set (see module docstring for semantics)."""
    files = _label_files(storage, prefix)
    unreadable: list[str] = []
    detections = 0
    by_category: dict[str, int] = {}
    sources: dict[str, int] = {}
    schema_versions: dict[str, int] = {}
    with_mask = 0
    fully_masked = 0
    attempted = 0
    legible = 0
    for stem, uri in files.items():
        try:
            data = json.loads(storage.read_bytes(uri).decode())
            img = ImageLabels.model_validate(data)
        except Exception:  # noqa: BLE001 — stats is a query; count and move on
            unreadable.append(stem)
            continue
        # from the RAW json: the pydantic default masks a missing field as "1"
        ver = str(data.get("schema_version", "absent")) if isinstance(data, dict) else "absent"
        schema_versions[ver] = schema_versions.get(ver, 0) + 1
        detections += len(img.detections)
        if all(d.mask is not None for d in img.detections):
            fully_masked += 1  # vacuously true for an empty file: nothing left to segment
        for d in img.detections:
            by_category[d.category] = by_category.get(d.category, 0) + 1
            src = d.source if d.source is not None else "none"
            sources[src] = sources.get(src, 0) + 1
            if d.mask is not None:
                with_mask += 1
            if d.text is not None:
                attempted += 1
                if d.text != "":
                    legible += 1
    return {
        "prefix": prefix,
        "files": len(files),
        "unreadable": len(unreadable),
        "detections": detections,
        "by_category": dict(sorted(by_category.items())),
        "masks": {
            "detections_with_mask": with_mask,
            "files_fully_masked": fully_masked,
            "coverage": round(with_mask / detections, 4) if detections else 0.0,
        },
        "text": {
            "attempted": attempted,
            "legible": legible,
            "coverage": round(attempted / detections, 4) if detections else 0.0,
        },
        "sources": dict(sorted(sources.items())),
        "schema_versions": dict(sorted(schema_versions.items())),
    }


def set_validate(prefix: str, *, storage: Storage) -> dict:
    """Schema-validate every file in the set. Returns {files, valid,
    violations: [{stem, error}]} — the CLI decides rc from len(violations)."""
    files = _label_files(storage, prefix)
    violations: list[dict] = []
    for stem, uri in files.items():
        try:
            ImageLabels.model_validate_json(storage.read_bytes(uri).decode())
        except Exception as exc:  # noqa: BLE001 — every failure IS the finding
            violations.append({"stem": stem, "error": str(exc).split("\n", 1)[0]})
    return {"files": len(files), "valid": len(files) - len(violations), "violations": violations}


def _normalized_detections(raw: bytes):
    """Detections in a canonical order, or None when unreadable. Detection
    CONTENT (bbox, category, score, source, mask, text) all participates;
    file-level image_path/width/height deliberately don't."""
    try:
        img = ImageLabels.model_validate_json(raw.decode())
    except Exception:  # noqa: BLE001
        return None
    dets = [d.model_dump() for d in img.detections]
    return sorted(dets, key=lambda d: (d["category"], d["bbox"]["x1"], d["bbox"]["y1"],
                                       d["bbox"]["x2"], d["bbox"]["y2"],
                                       json.dumps(d, sort_keys=True)))


def set_diff(prefix_a: str, prefix_b: str, *, storage_a: Storage, storage_b: Storage) -> dict:
    """Compare two label sets by stem. Lists carry the actionable stems
    (only_in_a / only_in_b / changed, sorted); identical and byte_identical are
    counts. byte_identical <= identical normally; an unreadable pair can only
    ever be byte-identical."""
    a = _label_files(storage_a, prefix_a)
    b = _label_files(storage_b, prefix_b)
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    changed: list[str] = []
    identical = 0
    byte_identical = 0
    for stem in sorted(set(a) & set(b)):
        raw_a = storage_a.read_bytes(a[stem])
        raw_b = storage_b.read_bytes(b[stem])
        if hashlib.sha256(raw_a).digest() == hashlib.sha256(raw_b).digest():
            byte_identical += 1
            identical += 1
            continue
        na, nb = _normalized_detections(raw_a), _normalized_detections(raw_b)
        if na is not None and na == nb:
            identical += 1
        else:
            changed.append(stem)
    return {
        "only_in_a": only_a,
        "only_in_b": only_b,
        "changed": changed,
        "identical": identical,
        "byte_identical": byte_identical,
    }
