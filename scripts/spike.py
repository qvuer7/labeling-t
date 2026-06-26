#!/usr/bin/env python
"""Stage-0 spike for any registered model (labeling_t/models.py).

Picks the model by key; reads only its endpoint from .env. Behavior (prompt,
coord space, parser) comes from the ModelSpec.

    uv run python scripts/spike.py --model qwen3_vl --check   # endpoint up?
    uv run python scripts/spike.py --model qwen3_vl --raw     # dump raw output
    uv run python scripts/spike.py --model qwen3_vl --images data/frames --out labels/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labeling_t.config import load_env  # noqa: E402
from labeling_t.model_client import VLLMClient  # noqa: E402
from labeling_t.models import get_spec  # noqa: E402
from labeling_t.prelabel import prelabel  # noqa: E402

DEFAULT_FRAMES = "data/spike_frames"
_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def _frames(directory: str) -> list[str]:
    d = Path(directory)
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
        return []
    return sorted(str(p) for e in _EXTS for p in d.glob(e))


def _csv(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def cmd_check(spec) -> int:
    endpoint = spec.endpoint_from_env()
    print(f"model: {spec.name}  ({spec.env_prefix}_ENDPOINT={endpoint or '(unset)'})")
    if not endpoint:
        return 1
    headers = {"Authorization": f"Bearer {spec.api_key_from_env()}"} if spec.api_key_from_env() else {}
    try:
        r = httpx.get(f"{endpoint}/v1/models", headers=headers, timeout=15)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"UNREACHABLE: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ids = [m.get("id") for m in r.json().get("data", [])]
    print(f"server UP. served ids: {ids}")
    if spec.name not in ids:
        print(f"  note: spec name {spec.name!r} not served — set --served-model-name {spec.name}")
    return 0


def cmd_raw(spec, images, categories, limit) -> int:
    if not images:
        print(f"no frames in {DEFAULT_FRAMES}/ — drop ~10 images there first")
        return 1
    client = VLLMClient.from_env(spec, categories=categories or None)
    print(f"model={spec.name} categories={client.categories} coord_space={spec.coord_space}\n")
    with client:
        for path in images[:limit]:
            print("=" * 70)
            print(f"FRAME: {path}")
            try:
                raw = client.infer(path)
            except Exception as exc:  # noqa: BLE001
                print(f"  request FAILED: {type(exc).__name__}: {exc}")
                continue
            print("--- RAW MODEL OUTPUT (paste this back) ---")
            print(raw)
            print("--- spec.parse() attempt ---")
            try:
                boxes = spec.parse(raw)
                print(f"  parsed {len(boxes)} boxes: {boxes[:3]}{' ...' if len(boxes) > 3 else ''}")
            except Exception as exc:  # noqa: BLE001
                print(f"  parse FAILED ({type(exc).__name__}: {exc}) — will tune spec.parse")
    print("=" * 70)
    return 0


def cmd_run(spec, images, categories, out, concurrency, min_score) -> int:
    if not images:
        print("no frames to label")
        return 1
    client = VLLMClient.from_env(spec, categories=categories or None)
    with client:
        labels = prelabel(images, client, out, min_score=min_score, max_concurrency=concurrency)
    print(f"labeled {len(labels)}/{len(images)} -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_env()
    p = argparse.ArgumentParser(description="model spike / stage-0 labeling")
    p.add_argument("--model", default="qwen3_vl", help="model spec key (labeling_t/models.py)")
    p.add_argument("--images", default=DEFAULT_FRAMES)
    p.add_argument("--out", default="labels")
    p.add_argument("--categories", type=_csv, default=None)
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--check", action="store_true")
    p.add_argument("--raw", action="store_true", help="dump raw output (T0 spike)")
    a = p.parse_args(argv)

    spec = get_spec(a.model)
    cats = a.categories or list(spec.categories)
    if a.check:
        return cmd_check(spec)
    if a.raw:
        return cmd_raw(spec, _frames(a.images), cats, a.limit)
    return cmd_run(spec, _frames(a.images), cats, a.out, a.concurrency, a.min_score)


if __name__ == "__main__":
    raise SystemExit(main())
