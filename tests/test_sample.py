"""Frame curation: classify corrupt/dark/flat/good, stratified-sample + copy."""

import io

from PIL import Image

from labeling_t.layout import DatasetLayout
from labeling_t.sample import curate
from labeling_t.storage import LocalStorage


def _jpg(img) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=95)
    return buf.getvalue()


def _gradient(w=120, h=120):
    im = Image.new("L", (w, h))
    im.putdata([x * 255 // w for _ in range(h) for x in range(w)])  # high-contrast ramp
    return im


def _seed(tmp_path):
    st = LocalStorage()
    base = str(tmp_path)
    lo = DatasetLayout("d", base=base)
    st.write_bytes(f"{lo.frames('g')}/g_00000.jpg", _jpg(_gradient()))                       # good
    st.write_bytes(f"{lo.frames('g')}/g_00001.jpg", _jpg(_gradient()))                       # good
    st.write_bytes(f"{lo.frames('g')}/g_00002.jpg", _jpg(Image.new("RGB", (120, 120), (8, 8, 8))))      # dark
    st.write_bytes(f"{lo.frames('g')}/g_00003.jpg", _jpg(Image.new("RGB", (120, 120), (130, 130, 130))))  # flat
    st.write_bytes(f"{lo.frames('g')}/g_00004.jpg", b"not-a-jpeg")                            # corrupt
    return st, base


def test_dry_run_classifies(tmp_path):
    st, base = _seed(tmp_path)
    rep = curate("d", "", 10, base=base, storage=st, dry_run=True)
    assert rep["total"] == 5
    assert rep["good"] >= 2
    assert rep["too_dark"] >= 1
    assert rep["too_flat"] >= 1
    assert rep["corrupt"] == 1


def test_real_sample_copies_only_good(tmp_path):
    st, base = _seed(tmp_path)
    rep = curate("d", "d-sample", 2, base=base, storage=st)
    assert rep["selected"] == 2
    out = DatasetLayout("d-sample", base=base)
    copied = [c for c in st.list(out.frames() + "/") if c.endswith(".jpg")]
    assert len(copied) == 2  # only the 2 good frames, dark/flat/corrupt excluded
