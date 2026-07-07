"""Per-model wrappers — each model's identity lives here, in code, not in .env.

A ModelSpec bundles everything that is INTRINSIC to a model: its served name,
the prompt phrasing it expects, how it reports box coordinates, and how to parse
its raw output. The framework stays decoupled from where the model is hosted:
the endpoint resolves at run time (podstate.resolve_endpoint — recorded pod /
--endpoint flag / SaaS default), keys come from the environment; never the
model's behavior.

    ModelSpec (code)            .env (secrets)              .labeling-t/pods.json (runtime)
      name, prompt              LOCATE_ANYTHING_API_KEY=…   endpoint of the running pod
      coord_space, parse
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
    # Which serving backend hosts this model. The first three all speak the
    # OpenAI chat protocol, so ChatClient drives them all; only the endpoint and
    # any quirk params differ (carried below). "transformers" is the odd one out.
    #   "vllm"         -> a vLLM OpenAI chat endpoint we rent (text out, spec.parse -> boxes)
    #   "openai"       -> OpenAI's hosted API (text out, spec.parse -> boxes)
    #   "gemini"       -> Google Gemini via its OpenAI-compat endpoint (same)
    #   "transformers" -> our FastAPI model-server (structured /infer; boxes already abs-px xyxy)
    backend: str = "vllm"
    # SaaS APIs live at a fixed, known URL, unlike a rented GPU whose host is only
    # known at run time. So a provider model bakes its base URL in here; .env need
    # only carry the key. {PREFIX}_ENDPOINT still overrides (Azure, a proxy, ...).
    default_endpoint: str = ""
    # OpenAI-compat chat route, appended to the (base) endpoint. vLLM/OpenAI sit at
    # /v1/chat/completions; Gemini's compat layer puts /v1beta/openai in the base,
    # so its path is just /chat/completions.
    chat_path: str = "/v1/chat/completions"
    # Non-standard chat-payload knobs sent verbatim (merged over the base body):
    # vLLM's repetition_penalty, a provider's response_format, etc. Lives with the
    # model because it's intrinsic to talking to THAT model, not to infra.
    extra_body: dict = field(default_factory=dict)
    # OpenAI image-detail hint ("low"/"high"; "" = provider default). "low" caps
    # the image at 512px and slashes image tokens — right for small crops (OCR),
    # wrong for full frames where boxes need resolution.
    image_detail: str = ""
    # Serving recipe (used by runpod.py to stand the model up):
    hf_model: str = ""                    # HF repo to serve, e.g. Qwen/Qwen3-VL-8B-Instruct
    serve_args: str = ""                  # extra vllm args (max-model-len, etc.)

    def endpoint_from_env(self) -> str:
        # env override wins; else the spec's baked-in default (SaaS provider URL).
        return (os.environ.get(f"{self.env_prefix}_ENDPOINT") or self.default_endpoint).rstrip("/")

    def api_key_from_env(self) -> str | None:
        return os.environ.get(f"{self.env_prefix}_API_KEY") or None


# --- registered models --------------------------------------------------------

# NVIDIA LocateAnything-3B: a generative grounding VLM (MoonViT + Qwen2.5-3B).
# Its custom arch isn't servable on stock vllm-openai, so it runs on OUR
# transformers model-server (backend="transformers"), same as OWLv2. The server's
# LocateAnythingAdapter owns the model-specific bits: it builds the model's own
# detection prompt, generates, and parses the `<box><x1><y1><x2><y2></box>` tokens
# (0-1000 normalized) into ABSOLUTE pixels — so coord_space="abs" and parse is
# unused here (the client reads structured boxes from JSON, never text).
LOCATE_ANYTHING = ModelSpec(
    key="locate_anything",
    name="locate-anything-3b",
    env_prefix="LOCATE_ANYTHING",
    prompt="",                            # the adapter applies the model's own template; queries sent structured
    coord_space="abs",
    categories=("player", "ball", "referee"),
    backend="transformers",
    hf_model="nvidia/LocateAnything-3B",
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
    # Grounding VLMs greedily loop, repeating one box to the token cap; vLLM honors
    # repetition_penalty on its OpenAI route and stops that. (Hosted OpenAI/Gemini
    # reject this param, which is exactly why it lives per-spec, not in the client.)
    extra_body={"repetition_penalty": 1.1},
    hf_model="Qwen/Qwen3-VL-8B-Instruct",
    # 8B + vision tower on a 24GB 4090: cap context so the KV cache fits.
    serve_args="--max-model-len 8192 --gpu-memory-utilization 0.95",
)

# --- hosted API models (no GPU to rent; ChatClient talks straight to the vendor) --

# Shared box-extraction prompt for the general-purpose VLMs (GPT/Gemini). They are
# not dedicated grounding models, so we ask explicitly for the same JSON shape
# parse_boxes already understands and let coord_space normalize.
_API_DETECT_PROMPT = (
    "Detect every object matching these categories: {categories}. "
    'Return ONLY a JSON list; each item {{"bbox_2d": [x1, y1, x2, y2], "label": "<category>"}}, '
    "with x1,y1,x2,y2 in pixel coordinates of THIS image (top-left origin). "
    "No prose, no code fences."
)

# OpenAI hosted vision (GPT-4o et al.). The OpenAI chat API IS what vLLM emulates,
# so ChatClient needs nothing new — only the baked-in base URL and a current model
# name. Swap `name` to retarget (gpt-4.1, o-series, ...); nothing else changes.
# NOTE: GPT pixel-grounding is approximate; confirm coord_space with a spike on a
# few frames (as was done for Qwen3-VL) before trusting box accuracy.
OPENAI_VL = ModelSpec(
    key="openai_vl",
    name="gpt-4o",
    env_prefix="OPENAI",
    backend="openai",
    default_endpoint="https://api.openai.com/v1",
    chat_path="/chat/completions",          # base already ends in /v1
    prompt=_API_DETECT_PROMPT,
    coord_space="abs",                      # we ask for pixels; verify per spike
    categories=("player", "ball", "referee"),
)

# Google Gemini via its OpenAI-compatibility endpoint — same wire shape, so the
# same ChatClient. Reads GEMINI_API_KEY. Swap `name` to retarget (gemini-2.5-pro).
# NOTE: Gemini's NATIVE grounding emits 0-1000 coords in [ymin,xmin,ymax,xmax]
# order; here we prompt for pixel [x1,y1,x2,y2] instead, so a spike must confirm
# both the space AND the axis order it actually returns before trusting boxes.
GEMINI_VL = ModelSpec(
    key="gemini_vl",
    name="gemini-2.5-flash",
    env_prefix="GEMINI",
    backend="gemini",
    default_endpoint="https://generativelanguage.googleapis.com/v1beta/openai",
    chat_path="/chat/completions",          # base already ends in /v1beta/openai
    prompt=_API_DETECT_PROMPT,
    coord_space="abs",                      # we ask for pixels; verify per spike
    categories=("player", "ball", "referee"),
)

# --- OCR / transcription (second-stage, transcribe.py) -------------------------

# Read the text out of a cropped region. Task prompt, not a detection prompt: the
# transcribe stage sends one CROP per call and expects plain text back, so
# spec.parse (box-typed) is unused on this path — transcribe.clean_text tidies
# the reply instead. Deliberately brace-free: ChatClient.build_payload runs
# .format(categories=...) on every prompt, so stray { } would explode.
_API_OCR_PROMPT = (
    "Transcribe the text visible in this image exactly as it appears. "
    "Return ONLY the transcribed text - no explanation, no quotes, no code fences. "
    "If no text is legible, return an empty string."
)

# OpenAI hosted OCR. Same endpoint/auth as OPENAI_VL (shared env_prefix -> shared
# OPENAI_API_KEY); differs only in task prompt + a cheaper model — per-crop OCR is
# easy, so gpt-4o-mini reads digits at ~1/25th the cost of gpt-4o.
OPENAI_OCR = ModelSpec(
    key="openai_ocr",
    name="gpt-4o-mini",
    env_prefix="OPENAI",
    backend="openai",
    default_endpoint="https://api.openai.com/v1",
    chat_path="/chat/completions",          # base already ends in /v1
    prompt=_API_OCR_PROMPT,
    coord_space="abs",                      # no boxes on this task; kept for consistency
    # crops are <512px, so "low" loses nothing and cuts image tokens ~3x —
    # that's 3x the throughput under a TPM rate limit and 1/3 the cost.
    image_detail="low",
)

# Gemini hosted OCR via the OpenAI-compat layer — same client, same GEMINI_API_KEY
# as GEMINI_VL. Flash is plenty for crop transcription and the cheapest option.
GEMINI_OCR = ModelSpec(
    key="gemini_ocr",
    name="gemini-2.5-flash",
    env_prefix="GEMINI",
    backend="gemini",
    default_endpoint="https://generativelanguage.googleapis.com/v1beta/openai",
    chat_path="/chat/completions",          # base already ends in /v1beta/openai
    prompt=_API_OCR_PROMPT,
    coord_space="abs",
)

# OWLv2 (Google) open-vocab detector, served by OUR transformers model-server
# (backend="transformers"). The server returns structured detections in ABSOLUTE
# pixels, so coord_space="abs" and parse is unused on this path (the client builds
# boxes from the JSON, not from text). First model of the transformers backend.
OWLV2 = ModelSpec(
    key="owlv2",
    name="owlv2",
    env_prefix="OWLV2",
    prompt="",                            # queries (the category list) are sent structured, not as a prompt
    coord_space="abs",
    categories=("player", "ball", "referee"),
    backend="transformers",
    hf_model="google/owlv2-base-patch16-ensemble",
)

# SAM2 (Segment Anything 2, Meta) — the SEGMENTER, not a detector. It takes box
# prompts (from any detector above) and returns a MASK per box, served by OUR
# transformers model-server (backend="transformers"). Stage 2 of the two-stage
# pipeline. No text query/prompt and no default categories: the boxes (and their
# labels) are supplied per call in `params`, not asked for. Uses transformers'
# native Sam2 (plain torch, no custom CUDA build), so it shares the same image.
SAM2 = ModelSpec(
    key="sam2",
    name="sam2",
    env_prefix="SAM2",
    prompt="",
    coord_space="abs",                    # masks/boxes already in absolute pixels
    backend="transformers",
    hf_model="facebook/sam2.1-hiera-large",
)

REGISTRY: dict[str, ModelSpec] = {
    LOCATE_ANYTHING.key: LOCATE_ANYTHING,
    QWEN3_VL.key: QWEN3_VL,
    OWLV2.key: OWLV2,
    OPENAI_VL.key: OPENAI_VL,
    GEMINI_VL.key: GEMINI_VL,
    OPENAI_OCR.key: OPENAI_OCR,
    GEMINI_OCR.key: GEMINI_OCR,
    SAM2.key: SAM2,
}


def get_spec(key: str) -> ModelSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown model {key!r}; known: {sorted(REGISTRY)}") from None
