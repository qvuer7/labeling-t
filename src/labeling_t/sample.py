"""Frame curation — a data-prep stage, not core labeling.

Drop unusable frames (corrupted / too dark / too flat) and stratified-sample N
across groups into a NEW dataset, so the initial labeling set is small and clean.
Quality is judged with cheap, generic heuristics (decode-check + brightness +
contrast); the thresholds are the domain-specific knob.

Frames are copied server-side (no download) into datasets/<out>/frames/.
"""

from __future__ import annotations

import collections
import io
import random
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageStat

from .layout import DatasetLayout
from .manifest import build_manifest, load_manifest
from .storage import Storage, open_storage

_IMG = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _group(frames_root: str, uri: str) -> str:
    return uri[len(frames_root) + 1:].split("/")[0]


def _score(storage: Storage, uri: str) -> tuple[float, float] | None:
    """(brightness, contrast) on a 64x64 grayscale, or None if it won't decode."""
    try:
        im = Image.open(io.BytesIO(storage.read_bytes(uri))).convert("L").resize((64, 64))
        st = ImageStat.Stat(im)
        return st.mean[0], st.stddev[0]
    except Exception:
        return None


def _percentiles(values: list[float], ps=(5, 25, 50, 75, 95)) -> dict[int, float]:
    if not values:
        return {p: 0.0 for p in ps}
    s = sorted(values)
    return {p: round(s[min(len(s) - 1, int(p / 100 * len(s)))], 1) for p in ps}


def curate(
    dataset: str,
    out_dataset: str,
    n: int,
    *,
    base: str | None = None,
    storage: Storage | None = None,
    min_brightness: float = 30.0,
    min_contrast: float = 18.0,
    max_concurrency: int = 16,
    dry_run: bool = False,
    seed: int = 0,
) -> dict:
    src = DatasetLayout.from_env(dataset, base=base)
    storage = storage or open_storage(src.root)
    frames_root = src.frames()
    frames = [u for u in storage.list(frames_root + "/") if u.lower().endswith(_IMG)]

    scores: dict[str, tuple[float, float]] = {}
    corrupt: list[str] = []
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for uri, sc in zip(frames, pool.map(lambda u: _score(storage, u), frames)):
            (corrupt.append(uri) if sc is None else scores.__setitem__(uri, sc))

    good = [u for u, (b, c) in scores.items() if b >= min_brightness and c >= min_contrast]
    dark = [u for u, (b, c) in scores.items() if b < min_brightness]
    flat = [u for u, (b, c) in scores.items() if b >= min_brightness and c < min_contrast]

    report = {
        "total": len(frames), "good": len(good), "too_dark": len(dark),
        "too_flat": len(flat), "corrupt": len(corrupt),
        "brightness_pct": _percentiles([b for b, _ in scores.values()]),
        "contrast_pct": _percentiles([c for _, c in scores.values()]),
        "thresholds": {"min_brightness": min_brightness, "min_contrast": min_contrast},
    }
    if dry_run:
        return report

    # stratified per-group sample, proportional to each group's good count
    by_group: dict[str, list[str]] = collections.defaultdict(list)
    for u in good:
        by_group[_group(frames_root, u)].append(u)
    rng = random.Random(seed)
    selected: list[str] = []
    for lst in by_group.values():
        rng.shuffle(lst)
        selected += lst[: round(n * len(lst) / max(1, len(good)))]
    rng.shuffle(selected)
    if len(selected) > n:
        selected = selected[:n]
    elif len(selected) < n:
        rest = [u for u in good if u not in set(selected)]
        rng.shuffle(rest)
        selected += rest[: n - len(selected)]

    dst = DatasetLayout.from_env(out_dataset, base=base)
    for u in selected:
        g, name = _group(frames_root, u), u.rsplit("/", 1)[-1]
        storage.copy(u, f"{dst.frames(g)}/{name}")

    prev = load_manifest(dataset, base=base, storage=storage) or {}
    build_manifest(
        out_dataset, base=base, storage=storage,
        categories=prev.get("categories"), source=f"datasets/{dataset}/frames (curated)",
        stride=(prev.get("extraction") or {}).get("stride"), model=prev.get("model"),
    )
    report["selected"] = len(selected)
    report["out_dataset"] = out_dataset
    return report
