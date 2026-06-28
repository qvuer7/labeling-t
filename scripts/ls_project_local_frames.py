#!/usr/bin/env python
"""QUICK FIX (throwaway): make an LS project whose frames load from a static host.

Only needed while the canonical object store (DO Spaces `ml-cv-data`) can't return
CORS headers, so the hosted LS browser can't fetch presigned frames. This points
the LS tasks at a CORS-enabled / same-origin host (e.g. frames served by the LS
droplet's Caddy at https://<ls-host>/frames/...) INSTEAD of presigned URLs.

It is NOT a framework command on purpose: the cloud path (`labeling-t
import-ls-cloud`, presigned S3) stays the default. Once CORS is fixed on Spaces,
go back to that and delete this script. It only reuses the framework as a library
(the adapter's existing image_base_url support — same one the local import path
uses); nothing in src/ changes.

Convergence is preserved: the static URL keeps the frame's <group>/<stem>.jpg, so
`labeling-t from-ls-cloud` still re-joins verified labels to the canonical S3
frames by stem (see verify.pull_verified).

    set -a; . ../StreamScout/.env; set +a
    uv run python scripts/ls_project_local_frames.py \
        --dataset ipbl-basketball-1k --group all \
        --url https://165-245-251-248.nip.io --api-key <token> \
        --project ipbl-basketball-1k-local --categories player,ball,referee \
        --image-base-url https://165-245-251-248.nip.io/frames
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labeling_t.adapters.label_studio import import_to_label_studio  # noqa: E402
from labeling_t.config import load_env  # noqa: E402
from labeling_t.layout import DatasetLayout  # noqa: E402
from labeling_t.schema import ImageLabels  # noqa: E402
from labeling_t.storage import open_storage  # noqa: E402


def main(argv=None) -> int:
    load_env()
    p = argparse.ArgumentParser(description="custom: LS project with frames from a static host (CORS quick fix)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--url", default=os.environ.get("LS_URL"), required="LS_URL" not in os.environ,
                   help="hosted Label Studio base URL (default $LS_URL)")
    p.add_argument("--api-key", default=os.environ.get("LS_API_KEY"), required="LS_API_KEY" not in os.environ,
                   help="LS API token (default $LS_API_KEY)")
    p.add_argument("--project", required=True)
    p.add_argument("--categories", required=True, help="comma-separated")
    p.add_argument("--image-base-url", default=os.environ.get("LS_IMAGE_BASE_URL"),
                   required="LS_IMAGE_BASE_URL" not in os.environ,
                   help="static host serving frames, e.g. https://<ls-host>/frames (default $LS_IMAGE_BASE_URL)")
    p.add_argument("--base", default=None, help="storage root (default s3://$S3_BUCKET)")
    a = p.parse_args(argv)

    layout = DatasetLayout.from_env(a.dataset, base=a.base)
    labels_prefix = layout.labels(a.group)
    storage = open_storage(labels_prefix)
    uris = [u for u in storage.list(labels_prefix + "/") if u.endswith(".json")]
    if not uris:
        print(f"no labels under {labels_prefix} (run prelabel-cloud first)", file=sys.stderr)
        return 1
    images = [ImageLabels.model_validate_json(storage.read_bytes(u).decode()) for u in uris]

    project = import_to_label_studio(
        images, base_url=a.url, api_key=a.api_key, project_title=a.project,
        categories=[c.strip() for c in a.categories.split(",") if c.strip()],
        # map each canonical frame URI -> <base>/<group>/<stem>.jpg (strip up to frames/)
        image_base_url=a.image_base_url, image_root=layout.frames(),
    )
    print(f"imported {len(images)} tasks into LS project {project.id} ({a.url}) — frames from {a.image_base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
