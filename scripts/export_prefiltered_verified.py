"""THROWAWAY — export a task-id-bounded slice of an LS project as verified labels.

SUPERSEDED except for the id threshold: `from-ls-cloud --include-accepted
--accepted-from <selector>` now does the corrected+accepted pull in the
framework. Only the one-off id<THRESHOLD slicing rule (below) keeps this
script alive — use it solely while a project is PARTIALLY verified.

One-off rule (2026-07-02): in LS project 11, every task with id < THRESHOLD has
been human-verified — tasks WITH annotations were corrected, tasks without were
viewed and accepted as-is (their prediction IS the verified truth). The
framework's from-ls-cloud has no id filter (and shouldn't grow one for this),
hence this script. Edit CONFIG, run:  uv run python scripts/export_prefiltered_verified.py
"""

from labeling_t.adapters.label_studio import from_label_studio
from labeling_t.config import load_env
from labeling_t.storage import open_storage

import httpx
import os
from pathlib import Path

# --- CONFIG ---------------------------------------------------------------
PROJECT_ID = 11
THRESHOLD = 12166        # tasks with id < THRESHOLD are verified
OUT_PREFIX = "s3://ml-cv-data/datasets/ipbl-basketball-1k/pre-filtered-verified/all"
FRAMES_PREFIX = "s3://ml-cv-data/datasets/ipbl-basketball-1k/frames/all"
# accepted-as-is tasks: the LS full export carries prediction IDs, not bodies —
# but the prediction IS the source pre-label, so copy it straight from here.
# (labels-yolo-sam2 replaced labels-yolo-seg on 2026-07-02; project 11's
# predictions were imported from the sam2-refined set.)
SRC_LABELS_PREFIX = "s3://ml-cv-data/datasets/ipbl-basketball-1k/labels-yolo-sam2/all"
# ---------------------------------------------------------------------------

load_env()
# NOT verify.fetch_ls_export: that omits download_all_tasks, so LS returns only
# ANNOTATED tasks — here the un-annotated ones are the point (accepted as-is).
r = httpx.get(
    f"{os.environ['LS_URL'].rstrip('/')}/api/projects/{PROJECT_ID}/export",
    params={"exportType": "JSON", "download_all_tasks": "true"},
    headers={"Authorization": f"Token {os.environ['LS_API_KEY']}"},
    timeout=300,
)
r.raise_for_status()
export = r.json()
below = [t for t in export if t["id"] < THRESHOLD]
corrected = [t for t in below if t.get("annotations")]
accepted = [t for t in below if not t.get("annotations")]
print(f"{len(below)} verified tasks (< {THRESHOLD}): "
      f"{len(corrected)} human-corrected, {len(accepted)} accepted as predicted")

storage = open_storage(OUT_PREFIX)

# human-corrected: parse the LS annotations back into the neutral schema
labels = from_label_studio(corrected, result_source="annotations")
for img in labels:
    stem = Path(img.image_path.split("?")[0]).stem  # presigned URL -> frame stem
    canonical = img.model_copy(update={"image_path": f"{FRAMES_PREFIX}/{stem}.jpg"})
    storage.write_text(f"{OUT_PREFIX}/{stem}.json", canonical.model_dump_json())

# accepted as predicted: the source pre-label is the verified truth, copy it
copied = 0
for t in accepted:
    stem = Path(t["data"]["image"].split("?")[0]).stem
    storage.copy(f"{SRC_LABELS_PREFIX}/{stem}.json", f"{OUT_PREFIX}/{stem}.json")
    copied += 1

print(f"wrote {len(labels)} corrected + {copied} accepted-as-is -> {OUT_PREFIX}")
