"""Ingest a local folder of images into a dataset's frames on storage.

The image counterpart to frames.py (which ingests *videos*): take a directory of
already-extracted images and upload them as a dataset group, so loose image
folders enter the same storage-centric pipeline as video-derived frames. Generic
and reusable — no dataset-specific logic lives here.

Resume-safe: images already present under the destination are skipped, so a
re-run after an interrupted upload only sends what's missing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def find_images(src: str, exts: tuple[str, ...] = IMAGE_EXTS) -> list[Path]:
    """Top-level image files in `src`, sorted (a flat folder of frames)."""
    root = Path(src)
    if not root.is_dir():
        raise NotADirectoryError(f"{src} is not a directory")
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts)


def ingest_images(
    src: str,
    dest_prefix: str,
    *,
    storage,
    exts: tuple[str, ...] = IMAGE_EXTS,
    resume: bool = True,
    max_concurrency: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[int, int]:
    """Upload every image in `src` to `dest_prefix/<filename>`.

    Filenames are preserved (the stem is the join key across pipeline stages).
    Returns (uploaded, total) — total is how many images `src` held, uploaded is
    how many were actually sent (the rest were already present when resume=True).

    `on_progress(done, total)` fires after each image is handled (uploaded or
    skipped) — a UI/progress-bar hook; default None keeps the silent behavior.
    """
    dest_prefix = dest_prefix.rstrip("/")
    files = find_images(src, exts)
    existing = set(storage.list(dest_prefix + "/")) if resume else set()

    def work(p: Path) -> int:
        dest = f"{dest_prefix}/{p.name}"
        if dest in existing:
            return 0
        storage.write_bytes(dest, p.read_bytes())
        return 1

    uploaded = done = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for fut in as_completed(pool.submit(work, p) for p in files):
            uploaded += fut.result()
            done += 1
            if on_progress is not None:
                on_progress(done, len(files))
    return uploaded, len(files)
