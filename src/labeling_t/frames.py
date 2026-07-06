"""Video -> frames: extract keyframes from videos in storage, write back.

Keyframes (I-frames) are the full-picture frames in a compressed video; the
frames between them only store motion deltas. Pulling only keyframes is fast
(ffmpeg skips decoding the delta frames) and yields the cleanest images for
labeling. ffmpeg flag: `-skip_frame nokey`.

Per chunk: download from storage -> ffmpeg keyframes -> upload JPEGs back to
storage. Idempotent: a chunk whose frames already exist is skipped.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from .storage import Storage, open_storage

VIDEO_EXTS = (".ts", ".mp4", ".mkv", ".mov", ".webm", ".m4v")


def extract_keyframes(video_path: str, out_dir: str) -> list[str]:
    """Extract every keyframe of `video_path` as JPEGs into out_dir. Returns
    the sorted list of produced files. Fast: only keyframes are decoded."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-nostdin",
            "-skip_frame", "nokey", "-i", video_path,
            "-vsync", "vfr", "-q:v", "2",
            str(out / "kf_%05d.jpg"),
        ],
        check=True,
    )
    return sorted(str(p) for p in out.glob("kf_*.jpg"))


def frames_from_videos(
    video_prefix: str,
    out_prefix: str,
    *,
    storage: Storage | None = None,
    stride: int = 1,
    resume: bool = True,
) -> int:
    """Extract keyframes from every video under `video_prefix`, writing frames
    to `out_prefix/<chunk>_NNNNN.jpg`. `stride` keeps every Kth keyframe.
    Returns the number of frames written (or already present when resuming)."""
    storage = storage or open_storage(video_prefix)
    out_prefix = out_prefix.rstrip("/")
    videos = [u for u in storage.list(video_prefix) if u.lower().endswith(VIDEO_EXTS)]
    if not videos:
        print(f"no videos under {video_prefix}", file=sys.stderr)
        return 0

    total = 0
    for v in videos:
        stem = Path(v).stem
        if resume:
            existing = storage.list(f"{out_prefix}/{stem}_")
            if existing:
                print(f"skip {stem} ({len(existing)} frames already exist)", file=sys.stderr)
                total += len(existing)
                continue
        with tempfile.TemporaryDirectory() as td:
            local = str(Path(td) / Path(v).name)
            Path(local).write_bytes(storage.read_bytes(v))
            keyframes = extract_keyframes(local, str(Path(td) / "frames"))
            kept = keyframes[:: max(1, stride)]
            for i, kf in enumerate(kept):
                storage.write_bytes(f"{out_prefix}/{stem}_{i:05d}.jpg", Path(kf).read_bytes())
            print(f"{stem}: {len(keyframes)} keyframes -> kept {len(kept)} -> {out_prefix}/",
                  file=sys.stderr)
            total += len(kept)
    return total
