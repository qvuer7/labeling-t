"""Ingest a local image folder into a dataset group's frames (resume-safe)."""

from pathlib import Path

import pytest

from labeling_t.ingest import find_images, ingest_images
from labeling_t.storage import LocalStorage


def _seed(src: Path) -> None:
    src.mkdir()
    (src / "a.jpg").write_bytes(b"x")
    (src / "b.png").write_bytes(b"y")
    (src / "notes.txt").write_bytes(b"skip-me")  # non-image, must be ignored


def test_find_images_filters_by_extension(tmp_path):
    src = tmp_path / "dd1"
    _seed(src)
    names = [p.name for p in find_images(str(src))]
    assert names == ["a.jpg", "b.png"]


def test_ingest_uploads_images_only(tmp_path):
    src = tmp_path / "dd1"
    _seed(src)
    dest = str(tmp_path / "out" / "frames" / "all")
    st = LocalStorage()

    uploaded, total = ingest_images(str(src), dest, storage=st)
    assert (uploaded, total) == (2, 2)
    assert sorted(Path(p).name for p in st.list(dest + "/")) == ["a.jpg", "b.png"]


def test_ingest_resumes_and_skips_existing(tmp_path):
    src = tmp_path / "dd1"
    _seed(src)
    dest = str(tmp_path / "out" / "frames" / "all")
    st = LocalStorage()

    ingest_images(str(src), dest, storage=st)
    # second run: everything already there -> nothing re-uploaded
    uploaded, total = ingest_images(str(src), dest, storage=st)
    assert (uploaded, total) == (0, 2)


def test_ingest_bad_src(tmp_path):
    with pytest.raises(NotADirectoryError):
        ingest_images(str(tmp_path / "missing"), str(tmp_path / "out"), storage=LocalStorage())
