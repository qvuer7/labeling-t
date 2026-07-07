"""Second-stage transcription — fill `Detection.text` on existing labels.

Where prelabel PRODUCES labels (frames in, boxes out), this stage ENRICHES
them: it reads a label set, crops each detection whose category matches the
ask, sends the crop to a hosted VLM (ChatClient — OpenAI/Gemini), and writes
the transcription back onto the same Detection. Dataset-neutral: which regions
carry text is entirely the caller's `categories` filter.

    labels/<stem>.json ──filter by category──► crop bbox (+pad) ──PNG bytes──►
        ChatClient.infer ──clean_text──► Detection.text ──► same JSON, rewritten

Resume is PER-DETECTION, keyed on `text`: None = never attempted (todo),
"" = attempted and nothing legible (done — never re-billed). A file whose
target detections all have text is skipped without touching the frame.

Failures go to `transcribe_failures.jsonl` (a distinct name so it can't collide
with prelabel's failures.jsonl; `.jsonl` is invisible to every `*.json` label
glob) and do not stop the run.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from PIL import Image

from .prelabel import _WRITE_BACKOFF, _WRITE_RETRIES
from .schema import BBox, ImageLabels

FAILURES_NAME = "transcribe_failures.jsonl"


def clean_text(raw: str) -> str:
    """Tidy a VLM transcription reply: strip whitespace, code fences, and one
    layer of surrounding quotes. Kept deliberately light — the prompt already
    asks for bare text; this only absorbs the common decorations."""
    s = raw.strip()
    if s.startswith("```"):
        # ```[lang]\n ... \n``` -> inner text
        s = s.strip("`").strip()
        first, _, rest = s.partition("\n")
        # a lone first line like "text"/"json" is the fence's language tag
        if rest and len(first.split()) == 1:
            s = rest.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def _crop_png(im: Image.Image, bbox: BBox, *, pad: int) -> bytes:
    """Crop `bbox` (+pad, clamped to the image) out of `im` as PNG bytes.
    PNG because crops are tiny and digits shouldn't survive a second JPEG pass."""
    x1 = max(0, int(round(bbox.x1)) - pad)
    y1 = max(0, int(round(bbox.y1)) - pad)
    x2 = min(im.width, int(round(bbox.x2)) + pad)
    y2 = min(im.height, int(round(bbox.y2)) + pad)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate crop {x1, y1, x2, y2} for bbox "
                         f"{bbox.x1, bbox.y1, bbox.x2, bbox.y2} in {im.width}x{im.height}")
    buf = io.BytesIO()
    im.crop((x1, y1, x2, y2)).save(buf, format="PNG")
    return buf.getvalue()


def _targets(labels: ImageLabels, categories: set[str]) -> list:
    """The detections this run still has to transcribe."""
    return [d for d in labels.detections if d.category in categories and d.text is None]


def _fill_text(labels: ImageLabels, image_bytes: bytes, client, *,
               categories: set[str], pad: int) -> int:
    """Crop + transcribe every un-attempted matching detection in place.
    Returns how many detections were filled."""
    todo = _targets(labels, categories)
    if not todo:
        return 0
    im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    for det in todo:
        det.text = clean_text(client.infer(_crop_png(im, det.bbox, pad=pad)))
    return len(todo)


def transcribe(
    labels_dir: str | Path,
    client,
    *,
    categories: list[str],
    pad: int = 2,
    images_dir: str | Path | None = None,
    max_concurrency: int = 4,
    resume: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Transcribe matching detections across a local label directory, in place.

    Frames are resolved from each label's `image_path`; `images_dir` overrides
    the directory (frames moved, or labels came from from-ls with bare names).
    Returns the number of label FILES enriched this run; per-file failures go
    to labels_dir/transcribe_failures.jsonl and don't stop the run.

    Concurrency fans out over FILES (regions within a file stay sequential so
    the read-modify-write needs no lock); default 4 — hosted APIs rate-limit.
    """
    cats = set(categories)
    out = Path(labels_dir)
    paths = sorted(p for p in out.glob("*.json"))
    failures_path = out / FAILURES_NAME
    fail_lock = threading.Lock()

    def work(dest: Path) -> int:
        try:
            labels = ImageLabels.model_validate_json(dest.read_text())
            if resume and not _targets(labels, cats):
                return 0
            frame = Path(labels.image_path)
            if images_dir is not None:
                frame = Path(images_dir) / frame.name
            filled = _fill_text(labels, frame.read_bytes(), client,
                                categories=cats, pad=pad)
        except Exception as exc:  # noqa: BLE001 - resilience is the point
            with fail_lock:
                failures_path.open("a").write(
                    json.dumps({"labels": str(dest), "error": f"{type(exc).__name__}: {exc}"})
                    + "\n"
                )
            return 0
        if not filled:
            return 0
        dest.write_text(labels.model_dump_json())
        return 1

    enriched = done = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for fut in as_completed(pool.submit(work, p) for p in paths):
            enriched += fut.result()
            done += 1
            if on_progress is not None:
                on_progress(done, len(paths))
    return enriched


def transcribe_cloud(
    labels_prefix: str,
    client,
    *,
    storage,
    categories: list[str],
    pad: int = 2,
    to_prefix: str | None = None,
    stems: set[str] | None = None,
    max_concurrency: int = 4,
    resume: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Transcribe matching detections across a cloud label set.

    In place by default; `to_prefix` instead writes enriched copies to a second
    label set (the source stays untouched — resume then reads the copy first).
    The frame comes from each label's `image_path` (the storage URI that
    prelabel_cloud saved) via storage.read_bytes; crops are cut locally and
    sent base64 — the frame is downloaded once per file regardless of region
    count. `stems` restricts the run to those file stems (sample-first
    workflow); None = the whole set. Returns the number of label files
    enriched this run; failures are flushed once to
    `<out>/transcribe_failures.jsonl`.
    """
    cats = set(categories)
    src = labels_prefix.rstrip("/")
    out = to_prefix.rstrip("/") if to_prefix else src
    keys = sorted(k for k in storage.list(src + "/") if k.endswith(".json"))
    if stems is not None:
        keys = [k for k in keys if k.rsplit("/", 1)[-1][:-5] in stems]
    done_set = set(storage.list(out + "/")) if (resume and to_prefix) else set()
    fails: list[str] = []
    lock = threading.Lock()

    def work(key: str) -> int:
        name = key.rsplit("/", 1)[-1]
        dest = f"{out}/{name}"
        try:
            # resume story: with to_prefix, the enriched copy (if present) is the
            # state to continue from; in place, source IS the state.
            read_from = dest if dest in done_set else key
            labels = ImageLabels.model_validate_json(storage.read_bytes(read_from))
            if resume and not _targets(labels, cats):
                return 0
            filled = _fill_text(labels, storage.read_bytes(labels.image_path), client,
                                categories=cats, pad=pad)
            if not filled:
                return 0
            # write inside the resilience boundary, retried like prelabel_cloud's
            payload = labels.model_dump_json()
            for attempt in range(_WRITE_RETRIES):
                try:
                    storage.write_text(dest, payload)
                    break
                except Exception:  # noqa: BLE001
                    if attempt == _WRITE_RETRIES - 1:
                        raise
                    time.sleep(_WRITE_BACKOFF * (attempt + 1))
        except Exception as exc:  # noqa: BLE001 - resilience is the point
            with lock:
                fails.append(json.dumps({"labels": key, "error": f"{type(exc).__name__}: {exc}"}))
            return 0
        return 1

    enriched = done = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for fut in as_completed(pool.submit(work, k) for k in keys):
            enriched += fut.result()
            done += 1
            if on_progress is not None:
                on_progress(done, len(keys))
    if fails:
        storage.write_text(f"{out}/{FAILURES_NAME}", "\n".join(fails) + "\n")
        print(f"{len(fails)} files failed -> {out}/{FAILURES_NAME}", file=sys.stderr)
    return enriched
