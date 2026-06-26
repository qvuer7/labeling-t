"""Batch pre-labeling orchestration (T3a) + the model-output parser (T3b).

T3a (transport-agnostic, built + tested now): stream image paths, fire bounded
concurrent inference, parse, map categories, filter by score, write one JSON per
frame, record failures, resume by skipping frames already written.

T3b (`parse_boxes`): turn the model's raw text into (box, label, score). This is
the ONE model-specific function. The reference implementation below targets a
common grounding-VLM JSON shape:

    [{"bbox_2d": [x1, y1, x2, y2], "label": "player", "score": 0.9}, ...]

Adjust `parse_boxes` (and `coord_space`) to LocateAnything's ACTUAL output once
the T0 spike shows it. Nothing else in the pipeline changes.

Coordinate space: VLMs disagree. `coord_space` picks the interpretation:
  "abs"      -> numbers are already absolute pixels
  "norm"     -> normalized [0,1]  (multiply by W/H)         <- LocateAnything likely
  "norm1000" -> normalized 0..1000 (Qwen-style)
All conversion goes through geometry.py.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image

from .geometry import normalized_to_abs
from .schema import BBox, Detection, ImageLabels

CoordSpace = Literal["abs", "norm", "norm1000"]

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_BBOX_KEYS = ("bbox_2d", "bbox", "box")
_LABEL_KEYS = ("label", "category", "name")
_SCORE_KEYS = ("score", "confidence")
# Pulls well-formed {bbox + label [+ score]} objects out of text even when the
# whole blob is truncated mid-stream (grounding VLMs loop and hit the token cap).
_OBJ_RE = re.compile(
    r'\{\s*"(?:bbox_2d|bbox|box)"\s*:\s*\[([^\]]*)\]\s*,\s*'
    r'"(?:label|category|name)"\s*:\s*"([^"]*)"'
    r'(?:\s*,\s*"(?:score|confidence)"\s*:\s*([0-9.]+))?',
)


class SupportsInfer(Protocol):
    # `spec` is a ModelSpec (carries coord_space / parse / name). Duck-typed
    # here to avoid a prelabel <-> models import cycle.
    spec: object

    def infer(self, image_path: str | Path) -> str: ...


# --- T3b: the one model-specific function ------------------------------------

def parse_boxes(text: str) -> list[tuple[list[float], str, float | None]]:
    """Raw model text -> list of (box[4], label, score|None), deduplicated.

    Clean JSON is parsed strictly. If the JSON is truncated (the model looped
    and hit the token cap), fall back to regex-extracting the well-formed
    objects. Identical (box, label) pairs are dropped, which removes the
    repetition the model emits. Raises only when nothing parseable is found.
    """
    cleaned = _FENCE.sub("", text.strip())
    items: list
    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        items = _regex_items(cleaned)
        if not items:
            raise  # genuinely unparseable -> caller records the failure
    else:
        if not isinstance(loaded, list):
            raise ValueError(f"expected a JSON list, got {type(loaded).__name__}")
        items = loaded

    out: list[tuple[list[float], str, float | None]] = []
    seen: set[tuple] = set()
    for item in items:
        box = next((item[k] for k in _BBOX_KEYS if k in item), None)
        label = next((item[k] for k in _LABEL_KEYS if k in item), None)
        if box is None or label is None:
            raise ValueError(f"item missing box/label: {item!r}")
        if len(box) != 4:
            raise ValueError(f"box must have 4 numbers: {box!r}")
        fbox = [float(v) for v in box]
        key = (tuple(round(v, 1) for v in fbox), str(label))
        if key in seen:
            continue  # drop repeated/looped boxes
        seen.add(key)
        score = next((item[k] for k in _SCORE_KEYS if k in item), None)
        out.append((fbox, str(label), score))
    return out


def _regex_items(text: str) -> list[dict]:
    """Salvage bbox objects from truncated/looping output."""
    items: list[dict] = []
    for nums, label, score in _OBJ_RE.findall(text):
        parts = [p for p in nums.replace(" ", "").split(",") if p]
        if len(parts) != 4:
            continue
        try:
            box = [float(p) for p in parts]
        except ValueError:
            continue
        d: dict = {"bbox_2d": box, "label": label}
        if score:
            d["score"] = float(score)
        items.append(d)
    return items


# --- T3a: orchestration -------------------------------------------------------

def _box_to_bbox(raw: list[float], w: int, h: int, coord_space: CoordSpace) -> BBox:
    x1, y1, x2, y2 = raw
    if coord_space == "abs":
        box = BBox.model_construct(x1=x1, y1=y1, x2=x2, y2=y2)
    elif coord_space == "norm":
        box = normalized_to_abs(x1, y1, x2, y2, w, h, scale=1.0)
    elif coord_space == "norm1000":
        box = normalized_to_abs(x1, y1, x2, y2, w, h, scale=1000.0)
    else:  # pragma: no cover - exhaustive
        raise ValueError(f"unknown coord_space {coord_space!r}")
    # Models drift slightly out of frame; clamp so the box is valid, not dropped.
    cx1, cy1 = max(0.0, min(box.x1, w)), max(0.0, min(box.y1, h))
    cx2, cy2 = max(0.0, min(box.x2, w)), max(0.0, min(box.y2, h))
    return BBox(x1=cx1, y1=cy1, x2=max(cx1, cx2), y2=max(cy1, cy2))


def _detections(
    raw: str, w: int, h: int, spec,
    *, category_map: dict[str, str] | None, min_score: float, strict_categories: bool,
) -> list[Detection]:
    """Model text + image dims -> Detections. Shared by the local and cloud
    paths; only how `raw`/`w`/`h` are obtained differs between them."""
    out: list[Detection] = []
    for box, label, score in spec.parse(raw):
        category = label if category_map is None else category_map.get(label)
        if category is None:
            if strict_categories:
                raise ValueError(f"unmapped category {label!r} from model")
            continue  # drop on purpose
        if score is not None and score < min_score:
            continue
        out.append(
            Detection(
                bbox=_box_to_bbox(box, w, h, spec.coord_space),
                category=category,
                score=score,
                source=spec.name,
            )
        )
    return out


def _label_one(
    image_path: str,
    client: SupportsInfer,
    *,
    category_map: dict[str, str] | None,
    min_score: float,
    strict_categories: bool,
) -> ImageLabels:
    spec = client.spec
    with Image.open(image_path) as im:  # local file -> dims from disk
        w, h = im.size
    raw = client.infer(image_path)
    dets = _detections(raw, w, h, spec, category_map=category_map,
                       min_score=min_score, strict_categories=strict_categories)
    return ImageLabels(image_path=image_path, width=w, height=h, detections=dets)


def prelabel(
    image_paths: list[str],
    client: SupportsInfer,
    out_dir: str | Path,
    *,
    category_map: dict[str, str] | None = None,
    min_score: float = 0.0,
    strict_categories: bool = False,
    max_concurrency: int = 8,
    resume: bool = True,
) -> list[ImageLabels]:
    """Pre-label every image. Writes one `<stem>.json` per frame into out_dir
    (its existence is the done-set, so re-running resumes). Per-frame failures
    go to out_dir/failures.jsonl and do not stop the run.

    Returns the labels produced THIS run (skipped-by-resume frames are reloaded
    from disk so the return is always the full set).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures_path = out / "failures.jsonl"
    fail_lock = threading.Lock()

    def out_path(p: str) -> Path:
        return out / (Path(p).stem + ".json")

    def work(p: str) -> ImageLabels | None:
        dest = out_path(p)
        if resume and dest.exists():
            return ImageLabels.model_validate_json(dest.read_text())
        try:
            labels = _label_one(
                p,
                client,
                category_map=category_map,
                min_score=min_score,
                strict_categories=strict_categories,
            )
        except Exception as exc:  # noqa: BLE001 - resilience is the point
            with fail_lock:
                failures_path.open("a").write(
                    json.dumps({"image": p, "error": f"{type(exc).__name__}: {exc}"})
                    + "\n"
                )
            return None
        dest.write_text(labels.model_dump_json())
        return labels

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        results = list(pool.map(work, image_paths))
    return [r for r in results if r is not None]


def prelabel_cloud(
    frame_uris: list[str],
    client: SupportsInfer,
    out_prefix: str,
    *,
    storage,
    category_map: dict[str, str] | None = None,
    min_score: float = 0.0,
    strict_categories: bool = False,
    max_concurrency: int = 8,
    resume: bool = True,
) -> int:
    """Pre-label frames living in object storage, fully in the cloud.

    Per frame: presign the URL (the GPU fetches it — no host download), read
    dims via a ranged header read, infer, and write a neutral-schema label JSON
    to `out_prefix/<stem>.json`. The saved `image_path` is the frame's storage
    URI. Returns the number of labels present (written + skipped-existing);
    per-frame failures go to `out_prefix/failures.jsonl` and don't stop the run.
    """
    spec = client.spec
    out_prefix = out_prefix.rstrip("/")
    existing = set(storage.list(out_prefix + "/")) if resume else set()
    fails: list[str] = []
    lock = threading.Lock()

    def work(uri: str) -> int:
        dest = f"{out_prefix}/{Path(uri).stem}.json"
        if dest in existing:
            return 1
        try:
            w, h = storage.image_size(uri)
            raw = client.infer(storage.presigned_url(uri))  # presigned -> GPU fetches
            dets = _detections(raw, w, h, spec, category_map=category_map,
                               min_score=min_score, strict_categories=strict_categories)
            labels = ImageLabels(image_path=uri, width=w, height=h, detections=dets)
        except Exception as exc:  # noqa: BLE001 - resilience is the point
            with lock:
                fails.append(json.dumps({"image": uri, "error": f"{type(exc).__name__}: {exc}"}))
            return 0
        storage.write_text(dest, labels.model_dump_json())
        return 1

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        written = sum(pool.map(work, frame_uris))
    if fails:
        storage.write_text(f"{out_prefix}/failures.jsonl", "\n".join(fails) + "\n")
        print(f"{len(fails)} frames failed -> {out_prefix}/failures.jsonl")
    return written
