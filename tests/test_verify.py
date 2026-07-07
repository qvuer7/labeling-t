"""verify.pull_verified: the LS pull-back, incl. --include-accepted semantics.

Pins the contracts that made project 11's export painful: LS only exports
annotated tasks unless download_all_tasks rides the request; accepted
(viewed-but-unsubmitted) tasks carry prediction IDs, not bodies, so their
source pre-label file is copied VERBATIM (byte-identity provenance); missing
sources are reported, never fatal.
"""

from __future__ import annotations

import json

import httpx
import pytest

import labeling_t.verify as verify
from labeling_t.schema import BBox, Detection, ImageLabels
from labeling_t.storage import LocalStorage


def _annotated_task(stem: str, *, w=100, h=100):
    """One LS task with a human box annotation (percent coords, per LS)."""
    return {
        "data": {"image": f"https://s3/frames/g/{stem}.jpg?X-Amz-Signature=abc"},
        "annotations": [{"result": [{
            "type": "rectanglelabels",
            "original_width": w, "original_height": h,
            "value": {"x": 10, "y": 10, "width": 30, "height": 30,
                      "rectanglelabels": ["player"]},
        }]}],
    }


def _viewed_task(stem: str):
    """A task the labeler accepted as-is: present in the full export, no annotation."""
    return {"data": {"image": f"https://s3/frames/g/{stem}.jpg?X-Amz-Signature=abc"},
            "annotations": []}


def _seed_source(base: str, stem: str) -> str:
    """A source pre-label file in labels-src/g (what fed the LS predictions)."""
    img = ImageLabels(image_path=f"{base}/datasets/d/frames/g/{stem}.jpg",
                      width=100, height=100,
                      detections=[Detection(bbox=BBox(x1=1, y1=1, x2=9, y2=9),
                                            category="player", source="model-x")])
    uri = f"{base}/datasets/d/labels-src/g/{stem}.json"
    LocalStorage().write_text(uri, img.model_dump_json())
    return uri


def test_fetch_export_param_assembly_default_vs_all_tasks(monkeypatch):
    seen = {}

    def fake_get(url, *, params, headers, timeout):
        seen.update(params=dict(params), url=url, timeout=timeout)
        return httpx.Response(200, json=[], request=httpx.Request("GET", url))

    monkeypatch.setattr(verify.httpx, "get", fake_get)
    verify.fetch_ls_export("http://ls", "k", 11)
    assert "download_all_tasks" not in seen["params"]
    verify.fetch_ls_export("http://ls", "k", 11, all_tasks=True)
    assert seen["params"]["download_all_tasks"] == "true"
    assert seen["timeout"] == 300  # full export is bigger; wait longer


def test_pull_verified_corrected_only_rewrites_image_path(tmp_path, monkeypatch):
    base = str(tmp_path)
    monkeypatch.setattr(verify, "fetch_ls_export",
                        lambda *a, **k: [_annotated_task("f1")])
    res = verify.pull_verified("d", "g", url="http://ls", api_key="k",
                               project_id=11, base=base)
    assert res == {"pulled": 1, "corrected": 1, "accepted": 0, "missing_source": []}
    out = ImageLabels.model_validate_json(
        (tmp_path / "datasets/d/verified/g/f1.json").read_text())
    # presigned URL replaced by the canonical frame URI (join-by-name survives)
    assert out.image_path == f"{base}/datasets/d/frames/g/f1.jpg"


def test_include_accepted_copies_source_byte_exact(tmp_path, monkeypatch):
    base = str(tmp_path)
    src_uri = _seed_source(base, "f2")
    export = [_annotated_task("f1"), _viewed_task("f2")]
    monkeypatch.setattr(verify, "fetch_ls_export", lambda *a, **k: export)
    res = verify.pull_verified("d", "g", url="http://ls", api_key="k", project_id=11,
                               base=base, include_accepted=True, accepted_from="labels-src")
    assert res == {"pulled": 2, "corrected": 1, "accepted": 1, "missing_source": []}
    # the accepted task's file is the SOURCE file, bit for bit (provenance intact)
    copied = (tmp_path / "datasets/d/verified/g/f2.json").read_bytes()
    assert copied == LocalStorage().read_bytes(src_uri)


def test_include_accepted_missing_source_reported_not_fatal(tmp_path, monkeypatch):
    base = str(tmp_path)
    _seed_source(base, "f2")
    export = [_viewed_task("f2"), _viewed_task("ghost")]
    monkeypatch.setattr(verify, "fetch_ls_export", lambda *a, **k: export)
    res = verify.pull_verified("d", "g", url="http://ls", api_key="k", project_id=11,
                               base=base, include_accepted=True, accepted_from="labels-src")
    assert res["accepted"] == 1 and res["missing_source"] == ["ghost"]
    assert res["pulled"] == 1
    assert not (tmp_path / "datasets/d/verified/g/ghost.json").exists()


def test_corrected_task_never_double_counted_as_accepted(tmp_path, monkeypatch):
    base = str(tmp_path)
    _seed_source(base, "f1")
    # full export lists the SAME task with its annotation; it must count once
    monkeypatch.setattr(verify, "fetch_ls_export", lambda *a, **k: [_annotated_task("f1")])
    res = verify.pull_verified("d", "g", url="http://ls", api_key="k", project_id=11,
                               base=base, include_accepted=True, accepted_from="labels-src")
    assert res == {"pulled": 1, "corrected": 1, "accepted": 0, "missing_source": []}
    # the corrected (human) version won, not the source copy
    out = ImageLabels.model_validate_json(
        (tmp_path / "datasets/d/verified/g/f1.json").read_text())
    assert out.detections[0].source is None  # human-verified, not "model-x"


def test_include_accepted_requires_accepted_from():
    with pytest.raises(ValueError, match="go together"):
        verify.pull_verified("d", "g", url="http://ls", api_key="k", project_id=11,
                             base="x", include_accepted=True)
    with pytest.raises(ValueError, match="go together"):
        verify.pull_verified("d", "g", url="http://ls", api_key="k", project_id=11,
                             base="x", accepted_from="labels-src")


def test_progress_covers_corrected_and_accepted(tmp_path, monkeypatch):
    base = str(tmp_path)
    _seed_source(base, "f2")
    export = [_annotated_task("f1"), _viewed_task("f2")]
    monkeypatch.setattr(verify, "fetch_ls_export", lambda *a, **k: export)
    calls = []
    verify.pull_verified("d", "g", url="http://ls", api_key="k", project_id=11,
                         base=base, include_accepted=True, accepted_from="labels-src",
                         on_progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 2), (2, 2)]
