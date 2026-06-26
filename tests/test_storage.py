"""Local storage backend (S3 is exercised live; this covers the offline path)."""

from PIL import Image

from labeling_t.storage import LocalStorage, is_s3, open_storage


def test_is_s3():
    assert is_s3("s3://bucket/key")
    assert not is_s3("data/frames/x.jpg")


def test_text_roundtrip(tmp_path):
    s = LocalStorage()
    uri = str(tmp_path / "sub" / "labels" / "x.json")
    s.write_text(uri, '{"a": 1}')  # creates parent dirs
    assert s.read_bytes(uri) == b'{"a": 1}'


def test_image_size(tmp_path):
    p = tmp_path / "f.jpg"
    Image.new("RGB", (640, 480), (0, 0, 0)).save(p)
    assert LocalStorage().image_size(str(p)) == (640, 480)


def test_presigned_url_is_passthrough_for_local(tmp_path):
    p = str(tmp_path / "f.jpg")
    assert LocalStorage().presigned_url(p) == p


def test_list_dir(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"y")
    got = LocalStorage().list(str(tmp_path))
    assert len(got) == 2 and all(g.endswith(".jpg") for g in got)


def test_open_storage_defaults_local_without_s3_env(monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    assert isinstance(open_storage("data/frames/x.jpg"), LocalStorage)
