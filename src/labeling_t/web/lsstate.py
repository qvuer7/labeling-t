"""Remember which Label Studio project backs each (dataset, group).

The import step creates an LS project and returns its id; the verify step needs
that id to export annotations back. We persist it as a small sidecar next to the
dataset (`<base>/datasets/<dataset>/ls_projects.json`) via the same Storage
backend, so the mapping survives a web-server restart and lives with the data.
"""

from __future__ import annotations

import json

from ..layout import DatasetLayout
from ..storage import open_storage

_FILE = "ls_projects.json"


def _path(dataset: str, base: str | None) -> str:
    return f"{DatasetLayout.from_env(dataset, base=base).root}/{_FILE}"


def load(dataset: str, *, base: str | None = None) -> dict[str, int]:
    """{group: project_id} for the dataset; empty if none recorded yet."""
    path = _path(dataset, base)
    try:
        return json.loads(open_storage(path).read_bytes(path).decode())
    except Exception:  # noqa: BLE001 - absent/unreadable -> nothing recorded
        return {}


def get_project_id(dataset: str, group: str, *, base: str | None = None) -> int | None:
    val = load(dataset, base=base).get(group)
    return int(val) if val is not None else None


def set_project_id(dataset: str, group: str, project_id: int, *, base: str | None = None) -> None:
    path = _path(dataset, base)
    data = load(dataset, base=base)
    data[group] = int(project_id)
    open_storage(path).write_text(path, json.dumps(data, indent=2))
