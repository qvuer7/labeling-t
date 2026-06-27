"""Dataset manifest: per-group counts from storage + declared-metadata merge."""

from labeling_t.layout import DatasetLayout
from labeling_t.manifest import build_manifest, load_manifest
from labeling_t.storage import LocalStorage


def _seed(tmp_path):
    st = LocalStorage()
    base = str(tmp_path)
    lo = DatasetLayout("d", base=base)
    st.write_bytes(f"{lo.frames('g1')}/g1_00000.jpg", b"x")
    st.write_bytes(f"{lo.frames('g1')}/g1_00001.jpg", b"x")
    st.write_bytes(f"{lo.frames('g2')}/g2_00000.jpg", b"x")
    st.write_text(f"{lo.labels('g1')}/g1_00000.json", "{}")
    return st, base


def test_counts_per_group_and_totals(tmp_path):
    st, base = _seed(tmp_path)
    m = build_manifest("d", base=base, storage=st, categories=["a", "b"], stride=3, model="qwen3-vl")
    assert m["totals"] == {"frames": 3, "labels": 1, "verified": 0, "groups": 2}
    assert m["groups"]["g1"] == {"frames": 2, "labels": 1, "verified": 0}
    assert m["groups"]["g2"] == {"frames": 1, "labels": 0, "verified": 0}
    assert m["categories"] == ["a", "b"]
    assert m["extraction"] == {"method": "keyframes", "stride": 3}
    assert m["model"] == "qwen3-vl"


def test_declared_metadata_persists_on_rebuild(tmp_path):
    st, base = _seed(tmp_path)
    build_manifest("d", base=base, storage=st, categories=["a", "b"], stride=3)
    # rebuild with no metadata args -> categories/extraction preserved, counts re-scanned
    m2 = build_manifest("d", base=base, storage=st)
    assert m2["categories"] == ["a", "b"]
    assert m2["extraction"] == {"method": "keyframes", "stride": 3}
    assert m2["totals"]["frames"] == 3


def test_load_manifest_roundtrip(tmp_path):
    st, base = _seed(tmp_path)
    assert load_manifest("d", base=base, storage=st) is None  # none yet
    build_manifest("d", base=base, storage=st, categories=["x"])
    assert load_manifest("d", base=base, storage=st)["categories"] == ["x"]
