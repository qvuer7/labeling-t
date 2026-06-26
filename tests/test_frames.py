"""Frame extraction orchestration (ffmpeg itself is stubbed; this covers the
download -> stride -> upload -> resume logic against LocalStorage)."""

from pathlib import Path

from labeling_t import frames as frames_mod
from labeling_t.frames import frames_from_videos
from labeling_t.storage import LocalStorage


def _fake_extract(n=5):
    def _impl(video_path, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(n):
            p = out / f"kf_{i:05d}.jpg"
            p.write_bytes(f"frame{i}".encode())
            paths.append(str(p))
        return sorted(paths)
    return _impl


def _make_video(tmp_path, name="cats_01.ts"):
    d = tmp_path / "videos"
    d.mkdir(exist_ok=True)
    (d / name).write_bytes(b"fake-ts-bytes")
    return str(d)


def test_keyframes_uploaded_with_stride(tmp_path, monkeypatch):
    monkeypatch.setattr(frames_mod, "extract_keyframes", _fake_extract(n=6))
    videos = _make_video(tmp_path)
    out = str(tmp_path / "frames")
    n = frames_from_videos(videos, out, storage=LocalStorage(), stride=2)
    # 6 keyframes, every 2nd -> 3 kept
    assert n == 3
    written = sorted(Path(out).glob("cats_01_*.jpg"))
    assert [p.name for p in written] == ["cats_01_00000.jpg", "cats_01_00001.jpg", "cats_01_00002.jpg"]


def test_resume_skips_done_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(frames_mod, "extract_keyframes", _fake_extract(n=4))
    videos = _make_video(tmp_path)
    out = str(tmp_path / "frames")
    frames_from_videos(videos, out, storage=LocalStorage())
    # second run must NOT call ffmpeg again
    monkeypatch.setattr(frames_mod, "extract_keyframes",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-extracted")))
    n = frames_from_videos(videos, out, storage=LocalStorage())
    assert n == 4  # reports existing


def test_non_video_files_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(frames_mod, "extract_keyframes", _fake_extract(n=2))
    d = tmp_path / "videos"
    d.mkdir()
    (d / "a.ts").write_bytes(b"x")
    (d / "notes.txt").write_bytes(b"x")
    n = frames_from_videos(str(d), str(tmp_path / "out"), storage=LocalStorage())
    assert n == 2  # only the .ts processed
