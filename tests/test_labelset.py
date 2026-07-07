"""labelset stats/validate/diff: the dataset-state primitives, over LocalStorage.

Pins the semantics that make these safe to act on: schema_version from RAW
json (not the pydantic default), the None/"" text resume contract, diff's
order-normalized + image_path-blind `changed` vs the byte_identical count
(the deletion-proof rule), and queries-never-fail on empty/corrupt input.
"""

from __future__ import annotations

import json

from labeling_t.cli import main
from labeling_t.labelset import set_diff, set_stats, set_validate
from labeling_t.schema import BBox, Detection, ImageLabels
from labeling_t.storage import LocalStorage

MASK = {"size": [100, 100], "counts": "b1"}


def _img(stem: str, *, dets: list[Detection], path: str | None = None) -> ImageLabels:
    return ImageLabels(image_path=path or f"frames/g/{stem}.jpg", width=100, height=100,
                       detections=dets)


def _det(cat: str = "player", *, x1: float = 10, mask: dict | None = None,
         text: str | None = None, source: str | None = "m1", score: float | None = 0.9) -> Detection:
    return Detection(bbox=BBox(x1=x1, y1=10, x2=x1 + 20, y2=40), category=cat,
                     score=score, source=source, mask=mask, text=text)


def _seed_set(tmp_path, name: str, files: dict[str, ImageLabels]) -> str:
    st = LocalStorage()
    prefix = str(tmp_path / name)
    for stem, img in files.items():
        st.write_text(f"{prefix}/{stem}.json", img.model_dump_json())
    return prefix


def test_stats_counts_detections_masks_text_sources(tmp_path):
    prefix = _seed_set(tmp_path, "labels", {
        "f1": _img("f1", dets=[_det("player", mask=MASK, text="42"),
                               _det("ball", x1=50, mask=MASK, text="")]),
        "f2": _img("f2", dets=[_det("player", source=None)]),
        "f3": _img("f3", dets=[]),
    })
    s = set_stats(prefix, storage=LocalStorage())
    assert s["files"] == 3 and s["unreadable"] == 0 and s["detections"] == 3
    assert s["by_category"] == {"ball": 1, "player": 2}
    assert s["sources"] == {"m1": 2, "none": 1}
    # masks: f1 fully masked, f3 vacuously (nothing left to segment), f2 not
    assert s["masks"] == {"detections_with_mask": 2, "files_fully_masked": 2, "coverage": 0.6667}
    # text resume contract: "" counts as attempted but not legible
    assert s["text"] == {"attempted": 2, "legible": 1, "coverage": 0.6667}
    assert s["schema_versions"] == {"1": 3}


def test_stats_schema_version_read_from_raw_json_and_unreadable_counted(tmp_path):
    st = LocalStorage()
    prefix = str(tmp_path / "labels")
    st.write_text(f"{prefix}/new.json", _img("new", dets=[]).model_dump_json())
    # pre-versioning file: field absent ON DISK (the pydantic default would lie)
    old = json.loads(_img("old", dets=[]).model_dump_json())
    del old["schema_version"]
    st.write_text(f"{prefix}/old.json", json.dumps(old))
    st.write_text(f"{prefix}/broken.json", "{not json")
    st.write_text(f"{prefix}/note.jsonl", "{}")  # failure sidecar: invisible
    s = set_stats(prefix, storage=LocalStorage())
    assert s["files"] == 3 and s["unreadable"] == 1
    assert s["schema_versions"] == {"1": 1, "absent": 1}


def test_stats_empty_prefix_is_a_query_not_an_error(tmp_path):
    s = set_stats(str(tmp_path / "nothing-here"), storage=LocalStorage())
    assert s["files"] == 0 and s["detections"] == 0


def test_validate_reports_violations(tmp_path):
    st = LocalStorage()
    prefix = str(tmp_path / "labels")
    st.write_text(f"{prefix}/good.json", _img("good", dets=[_det()]).model_dump_json())
    bad = json.loads(_img("bad", dets=[_det()]).model_dump_json())
    bad["detections"][0]["bbox"]["x2"] = 999  # exceeds width=100
    st.write_text(f"{prefix}/bad.json", json.dumps(bad))
    v = set_validate(prefix, storage=LocalStorage())
    assert v["files"] == 2 and v["valid"] == 1
    assert [x["stem"] for x in v["violations"]] == ["bad"]


def test_diff_rewritten_image_path_is_not_a_change(tmp_path):
    # the verified pull-back rewrites image_path; content-diff must stay quiet
    d = [_det("player"), _det("ball", x1=50)]
    pa = _seed_set(tmp_path, "a", {"f1": _img("f1", dets=d)})
    pb = _seed_set(tmp_path, "b", {"f1": _img("f1", dets=list(reversed(d)),
                                              path="s3://elsewhere/f1.jpg")})
    r = set_diff(pa, pb, storage_a=LocalStorage(), storage_b=LocalStorage())
    # different bytes, different order, different path -> still identical content
    assert r["identical"] == 1 and r["byte_identical"] == 0 and r["changed"] == []


def test_diff_partitions_stems(tmp_path):
    pa = _seed_set(tmp_path, "a", {
        "both_same": _img("x", dets=[_det()]),
        "both_changed": _img("y", dets=[_det("player")]),
        "a_only": _img("z", dets=[]),
    })
    pb = _seed_set(tmp_path, "b", {
        "both_same": _img("x", dets=[_det()]),
        "both_changed": _img("y", dets=[_det("referee")]),  # category changed
        "b_only": _img("w", dets=[]),
    })
    r = set_diff(pa, pb, storage_a=LocalStorage(), storage_b=LocalStorage())
    assert r["only_in_a"] == ["a_only"] and r["only_in_b"] == ["b_only"]
    assert r["changed"] == ["both_changed"]
    # byte-identical dump -> counts in BOTH identical and byte_identical
    assert r["identical"] == 1 and r["byte_identical"] == 1


def test_diff_detection_content_participates(tmp_path):
    # same box, mask added on one side -> changed (an enrichment IS a change)
    pa = _seed_set(tmp_path, "a", {"f": _img("f", dets=[_det()])})
    pb = _seed_set(tmp_path, "b", {"f": _img("f", dets=[_det(mask=MASK)])})
    r = set_diff(pa, pb, storage_a=LocalStorage(), storage_b=LocalStorage())
    assert r["changed"] == ["f"]


# ---- CLI wiring ---------------------------------------------------------------

def test_cli_stats_json_envelope(tmp_path, capsys):
    prefix = _seed_set(tmp_path, "labels", {"f1": _img("f1", dets=[_det(text="7")])})
    rc = main(["stats", "--labels", prefix, "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["ok"] is True
    assert envelope["result"]["files"] == 1 and envelope["result"]["detections"] == 1
    assert envelope["result"]["set"] == prefix


def test_cli_validate_rc1_and_truncation_on_dirty_set(tmp_path, capsys):
    st = LocalStorage()
    prefix = str(tmp_path / "labels")
    for i in range(3):
        st.write_text(f"{prefix}/bad{i}.json", "{not json")
    rc = main(["validate", "--labels", prefix, "--limit", "2", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 1 and envelope["ok"] is False
    assert envelope["result"]["violations_total"] == 3
    assert len(envelope["result"]["violations"]) == 2  # --limit caps the listing
    assert envelope["result"]["valid"] == 0


def test_cli_validate_clean_rc0(tmp_path, capsys):
    prefix = _seed_set(tmp_path, "labels", {"f1": _img("f1", dets=[])})
    rc = main(["validate", "--labels", prefix, "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["result"]["violations_total"] == 0


def test_cli_diff_with_selectors_uses_layout(tmp_path, capsys, monkeypatch):
    # dataset-coordinates mode: selectors resolve through DatasetLayout.set_prefix
    monkeypatch.delenv("S3_BUCKET", raising=False)
    base = str(tmp_path)
    st = LocalStorage()
    img = _img("f1", dets=[_det()])
    st.write_text(f"{base}/datasets/d/labels/g/f1.json", img.model_dump_json())
    st.write_text(f"{base}/datasets/d/verified/g/f1.json", img.model_dump_json())
    rc = main(["diff", "--dataset", "d", "--group", "g", "--a", "labels", "--b", "verified",
               "--base", base, "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["result"]["byte_identical"] == 1


def test_cli_stats_bad_selector_fails_loudly(tmp_path, capsys):
    rc = main(["stats", "--dataset", "d", "--group", "g", "--set", "frames",
               "--base", str(tmp_path), "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 1 and "invalid set selector" in envelope["error"]["message"]


def test_cli_stats_flat_set_empty_group(tmp_path, capsys):
    # a label set living directly under <root>/labels/ (no group folder) is
    # addressable with --group "" — real case: datasets/ipbl-basketball-seg
    base = str(tmp_path)
    LocalStorage().write_text(f"{base}/datasets/d/labels/f1.json",
                              _img("f1", dets=[_det()]).model_dump_json())
    rc = main(["stats", "--dataset", "d", "--group", "", "--set", "labels",
               "--base", base, "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["result"]["files"] == 1
