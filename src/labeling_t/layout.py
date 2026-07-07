"""Canonical storage layout for a labeling dataset.

The ONE place the folder structure is defined, so frame/label/export keys never
drift apart. Dataset-grouped: everything for one labeling effort is self-
contained under its dataset name.

    <base>/datasets/<dataset>/
        frames/<group>/<stem>.jpg                # keyframes
        labels/<group>/<stem>.json               # model pre-labels (neutral schema)
        verified/<group>/<stem>.json             # human-verified (neutral schema)
        export/<version>/annotations.coco.json   # exports

`group` is a dataset-neutral partition (e.g. a source video / game / clip);
keep dataset-specific vocabulary out of the framework surface.

`base` is a storage root — an s3:// URI (s3://ml-cv-data) or a local dir (data).
A frame, its pre-label, and its verified label share the same filename stem, so
they join by name across stages (no manifest needed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetLayout:
    dataset: str
    base: str = "s3://ml-cv-data"

    @classmethod
    def from_env(cls, dataset: str, base: str | None = None) -> "DatasetLayout":
        """Default base to s3://$S3_BUCKET when configured, else local 'data'."""
        if base is None:
            bucket = os.environ.get("S3_BUCKET", "").strip()
            base = f"s3://{bucket}" if bucket else "data"
        return cls(dataset=dataset, base=base.rstrip("/"))

    @property
    def root(self) -> str:
        return f"{self.base}/datasets/{self.dataset}"

    def frames(self, group: str = "") -> str:
        return f"{self.root}/frames/{group}".rstrip("/")

    def labels(self, group: str = "", name: str = "") -> str:
        """Pre-label prefix for `group`. `name` namespaces a non-default model's
        pre-labels into a sibling dir (labels-<name>/<group>) so several models'
        pre-labels coexist without clobbering — e.g. name="locateanything" keeps
        them apart from the default qwen labels/. Empty name = the default labels/."""
        leaf = f"labels-{name}" if name else "labels"
        return f"{self.root}/{leaf}/{group}".rstrip("/")

    def verified(self, group: str = "", name: str = "") -> str:
        """Human-verified prefix for `group`. `name` namespaces a second verified
        pass into a sibling dir (verified-<name>/<group>) — e.g. name="masks"
        keeps mask-verified labels apart from the box-verified verified/."""
        leaf = f"verified-{name}" if name else "verified"
        return f"{self.root}/{leaf}/{group}".rstrip("/")

    def set_prefix(self, group: str, selector: str) -> str:
        """Prefix for a label SET named by its storage leaf — the one selector
        vocabulary agents use to point a command at any set:

            labels | labels-<name> | verified | verified-<name>

        (i.e. exactly what `aws s3 ls` shows under the dataset root). Consumed
        by stats/validate/diff, --frames-from, --accepted-from, render.
        ValueError on anything else, so a typo'd set name fails loudly instead
        of scanning an empty prefix."""
        stage, dash, name = selector.partition("-")
        if stage not in ("labels", "verified") or (dash and not name):
            raise ValueError(
                f"invalid set selector {selector!r}: expected labels, labels-<name>, "
                "verified, or verified-<name>"
            )
        return f"{self.root}/{selector}/{group}".rstrip("/")

    def export(self, version: str) -> str:
        return f"{self.root}/export/{version}".rstrip("/")
