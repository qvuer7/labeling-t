"""Per-model wrappers — each model's identity lives here, in code, not in .env.

A ModelSpec bundles everything that is INTRINSIC to a model: its served name,
the prompt phrasing it expects, how it reports box coordinates, and how to parse
its raw output. The framework stays decoupled from where the model is hosted:
only the endpoint + key come from the environment (per model), never the
model's behavior.

    ModelSpec (code)            .env (infra only)
      name, prompt              LOCATE_ANYTHING_ENDPOINT=http://gpu:8000
      coord_space, parse        LOCATE_ANYTHING_API_KEY=...
      default categories

Add a model = add a ModelSpec instance + register it. Project-level knobs
(category_map, min_score) are NOT here — those belong to a labeling run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Literal

from .prelabel import parse_boxes  # the default reference parser

CoordSpace = Literal["abs", "norm", "norm1000"]
# raw model text -> [(box[4], label, score|None)]
ParseFn = Callable[[str], list[tuple[list[float], str, float | None]]]


@dataclass(frozen=True)
class ModelSpec:
    key: str                              # registry key, e.g. "locate_anything"
    name: str                             # vLLM --served-model-name
    prompt: str                           # has a {categories} placeholder
    env_prefix: str                       # reads {PREFIX}_ENDPOINT / {PREFIX}_API_KEY
    coord_space: CoordSpace = "norm"
    categories: tuple[str, ...] = ()      # default ask; overridable per run
    parse: ParseFn = parse_boxes
    # Serving recipe (used by scripts/runpod.py to stand the model up on vLLM):
    hf_model: str = ""                    # HF repo to serve, e.g. Qwen/Qwen3-VL-8B-Instruct
    serve_args: str = ""                  # extra vllm args (max-model-len, etc.)

    def endpoint_from_env(self) -> str:
        return os.environ.get(f"{self.env_prefix}_ENDPOINT", "").rstrip("/")

    def api_key_from_env(self) -> str | None:
        return os.environ.get(f"{self.env_prefix}_API_KEY") or None


# --- registered models --------------------------------------------------------

# NVIDIA LocateAnything-3B (Nemotron-family grounding VLM). Returns boxes only.
# coord_space + parse are the two things the T0 spike confirms against real
# output; adjust them here, nothing else changes.
# Output format (per HF model card): boxes as `<box><x1><y1><x2><y2></box>`
# special tokens, coords normalized 0-1000. Prompt is the model's own template.
# parse stays the JSON reference parser until the spike shows the exact text;
# then swap to a <box>-parser.
LOCATE_ANYTHING = ModelSpec(
    key="locate_anything",
    name="locate-anything-3b",
    env_prefix="LOCATE_ANYTHING",
    prompt="Locate all the instances that matches the following description: {categories}.",
    coord_space="norm1000",
    categories=("player", "ball"),
    hf_model="nvidia/LocateAnything-3B",  # NOTE: not servable on stock vllm-openai
    serve_args="--trust-remote-code --max-model-len 8192",
)

# Qwen3-VL: current-gen open grounding VLM, natively supported by vLLM (no
# trust-remote-code). Outputs JSON `{"bbox_2d":[x1,y1,x2,y2],"label":..}` in
# ABSOLUTE pixel coords, which the reference parse_boxes already handles.
# (LocateAnything-3B's custom arch isn't servable on stock vllm/vllm-openai;
# kept above for if/when a compatible image exists.)
QWEN3_VL = ModelSpec(
    key="qwen3_vl",
    name="qwen3-vl",
    env_prefix="QWEN_VL",
    prompt=(
        "Detect every object matching these categories: {categories}. "
        'Return ONLY a JSON list; each item {{"bbox_2d": [x1, y1, x2, y2], "label": "<category>"}}.'
    ),
    # Confirmed by T0 spike: Qwen3-VL emits coords normalized 0-1000, not pixels.
    coord_space="norm1000",
    categories=("player", "ball", "referee"),
    hf_model="Qwen/Qwen3-VL-8B-Instruct",
    # 8B + vision tower on a 24GB 4090: cap context so the KV cache fits.
    serve_args="--max-model-len 8192 --gpu-memory-utilization 0.95",
)

REGISTRY: dict[str, ModelSpec] = {
    LOCATE_ANYTHING.key: LOCATE_ANYTHING,
    QWEN3_VL.key: QWEN3_VL,
}


def get_spec(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {sorted(REGISTRY)}") from None
