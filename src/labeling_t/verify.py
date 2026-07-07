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


def fetch_ls_export(url: str, api_key: str, project_id: int | str, *,
                    all_tasks: bool = False) -> list[dict]:
    """GET a Label Studio project's export (JSON) straight from the API.

    By default LS exports only ANNOTATED tasks — viewed-but-unsubmitted ones
    are silently absent. all_tasks=True adds download_all_tasks so every task
    comes back (bigger payload, hence the longer timeout)."""
    params = {"exportType": "JSON"}
    if all_tasks:
        params["download_all_tasks"] = "true"
    r = httpx.get(
        f"{url.rstrip('/')}/api/projects/{project_id}/export",
        params=params,
        headers={"Authorization": f"Token {api_key}"},
        timeout=300 if all_tasks else 180,
    )
    r.raise_for_status()
    return r.json()


def _task_stem(task: dict) -> str:
    """A task's frame stem from its image URL (presigned query stripped)."""
    return Path(str(task.get("data", {}).get("image", "")).split("?")[0]).stem


def pull_verified(
    dataset: str,
    group: str,
    *,
    url: str,
    api_key: str,
    project_id: int | str,
    base: str | None = None,
    name: str = "",
    include_accepted: bool = False,
    accepted_from: str = "",
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Export verified annotations from LS and write them to `verified[-name]/<group>/`.

    Each corrected label's `image_path` is rewritten to the canonical frame URI
    (`frames/<group>/<stem>.jpg`) so verified labels join frames by name,
    regardless of the presigned URL the labeler's browser actually fetched.
    `name` namespaces a second verified pass (e.g. "masks") apart from the
    box-verified verified/.

    include_accepted=True (requires accepted_from, a set selector) also treats
    tasks WITHOUT an annotation as verified-by-viewing: the full export carries
    their prediction IDs but not bodies, and the source pre-label file IS that
    prediction — so each accepted task is a byte-exact storage.copy from
    `accepted_from` (provenance preserved; proven on LS project 11). Stems
    whose source file is missing are reported, not fatal.

    Returns {"pulled", "corrected", "accepted", "missing_source": [stems]}.
    """
    if include_accepted != bool(accepted_from):
        raise ValueError("--include-accepted and --accepted-from go together: accepted tasks "
                         "are copied from the label set that fed the LS project's predictions")
    layout = DatasetLayout.from_env(dataset, base=base)
    frames_prefix, verified_prefix = layout.frames(group), layout.verified(group, name)
    storage = open_storage(verified_prefix)

    export = fetch_ls_export(url, api_key, project_id, all_tasks=include_accepted)
    labels = from_label_studio(export, result_source="annotations")
    corrected_stems = {Path(img.image_path.split("?")[0]).stem for img in labels}
    accepted_stems: list[str] = []
    if include_accepted:
        accepted_stems = sorted({
            stem for t in export
            if (stem := _task_stem(t)) and stem not in corrected_stems
        })
    total = len(labels) + len(accepted_stems)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if on_progress is not None:
            on_progress(done, total)

    for img in labels:
        stem = Path(img.image_path.split("?")[0]).stem  # presigned URL -> frame stem
        canonical = img.model_copy(update={"image_path": f"{frames_prefix}/{stem}.jpg"})
        storage.write_text(f"{verified_prefix}/{stem}.json", canonical.model_dump_json())
        tick()

    missing: list[str] = []
    if accepted_stems:
        src_prefix = layout.set_prefix(group, accepted_from)
        available = {u.rsplit("/", 1)[-1][:-5]
                     for u in storage.list(src_prefix + "/") if u.endswith(".json")}
        for stem in accepted_stems:
            if stem in available:
                storage.copy(f"{src_prefix}/{stem}.json", f"{verified_prefix}/{stem}.json")
            else:
                missing.append(stem)
            tick()
    accepted = len(accepted_stems) - len(missing)
    return {"pulled": len(labels) + accepted, "corrected": len(labels),
            "accepted": accepted, "missing_source": missing}
