"""Pull human-verified annotations from Label Studio back into the dataset.

The closing stage of the loop: a labeler corrects the model's boxes in LS, and
this lifts that verified truth back into the neutral schema on storage
(`verified/<group>/<stem>.json`). Orchestration only — the LS-specific parsing
lives in adapters/label_studio.py; this module wires the LS API export to the
DatasetLayout + Storage.

Reused by both `labeling-t from-ls-cloud` (CLI) and the web UI's Verify step, so
the join-by-name rewrite (presigned URL -> canonical frame URI) has one home.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import httpx

from .adapters.label_studio import from_label_studio
from .layout import DatasetLayout
from .storage import open_storage


def fetch_ls_export(url: str, api_key: str, project_id: int | str) -> list[dict]:
    """GET a Label Studio project's export (JSON) straight from the API."""
    r = httpx.get(
        f"{url.rstrip('/')}/api/projects/{project_id}/export",
        params={"exportType": "JSON"},
        headers={"Authorization": f"Token {api_key}"},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def pull_verified(
    dataset: str,
    group: str,
    *,
    url: str,
    api_key: str,
    project_id: int | str,
    base: str | None = None,
    name: str = "",
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Export verified annotations from LS and write them to `verified[-name]/<group>/`.

    Each saved label's `image_path` is rewritten to the canonical frame URI
    (`frames/<group>/<stem>.jpg`) so verified labels join frames by name,
    regardless of the presigned URL the labeler's browser actually fetched.
    `name` namespaces a second verified pass (e.g. "masks") apart from the
    box-verified verified/. Returns the number of verified label files written.
    """
    layout = DatasetLayout.from_env(dataset, base=base)
    frames_prefix, verified_prefix = layout.frames(group), layout.verified(group, name)
    storage = open_storage(verified_prefix)

    labels = from_label_studio(fetch_ls_export(url, api_key, project_id),
                               result_source="annotations")
    total = len(labels)
    for i, img in enumerate(labels, 1):
        stem = Path(img.image_path.split("?")[0]).stem  # presigned URL -> frame stem
        canonical = img.model_copy(update={"image_path": f"{frames_prefix}/{stem}.jpg"})
        storage.write_text(f"{verified_prefix}/{stem}.json", canonical.model_dump_json())
        if on_progress is not None:
            on_progress(i, total)
    return total
