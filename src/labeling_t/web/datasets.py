"""Read-only dataset overview for the UI — datasets, groups, per-stage counts.

Reads the generated manifest.json (regenerable from storage); self-heals by
building one when absent. Adds the recorded Label Studio project ids so the UI
can link straight into LS.
"""

from __future__ import annotations

from ..layout import DatasetLayout
from ..manifest import build_manifest, load_manifest
from ..storage import open_storage
from . import lsstate


def _resolved_base(base: str | None) -> str:
    """The storage root the env resolves to (s3://$S3_BUCKET or local 'data')."""
    return DatasetLayout.from_env("_", base=base).base


def list_dataset_names(base: str | None = None) -> list[str]:
    prefix = f"{_resolved_base(base)}/datasets/"
    storage = open_storage(prefix)
    names = {
        key[len(prefix):].split("/")[0]
        for key in storage.list(prefix)
        if key.startswith(prefix) and "/" in key[len(prefix):]
    }
    return sorted(n for n in names if n)


def dataset_overview(dataset: str, *, base: str | None = None) -> dict:
    """Manifest (built if missing) + the LS project ids for the dataset."""
    manifest = load_manifest(dataset, base=base) or build_manifest(dataset, base=base)
    return {"manifest": manifest, "ls_projects": lsstate.load(dataset, base=base)}


def list_datasets(base: str | None = None) -> list[dict]:
    return [dataset_overview(n, base=base) for n in list_dataset_names(base)]
