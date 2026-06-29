"""NVIDIA LocateAnything-3B adapter — a generative grounding VLM on our server.

Unlike OWLv2 (a discriminative detector with per-box scores), LocateAnything is a
3B VLM (MoonViT + Qwen2.5-3B) that GENERATES a token sequence containing box
tokens. Its official output utility (`parse_boxes`, from the model card) extracts
only `<box><x1><y1><x2><y2></box>` runs in 0-1000 normalized coords and DROPS the
category text — there is no documented, stable box→label format for a single
multi-category pass.

So we query ONE category at a time and tag every box from that pass with that
query. This trades N forward passes for guaranteed-correct labels (the model is
fast: Parallel Box Decoding emits all boxes in one step, not autoregressively).
There are no confidence scores on this path, so `score=None`.

    queries=[player, ball]  ──►  per-query prompt ──► generate ──► "<box>..</box>.."
                                 _parse_boxes (0-1000) ──► _to_wire (abs-px, clamp)

`_parse_boxes` / `_to_wire` are pure (no torch) so the coord math is unit-tested
without a GPU; `detect()`/`load()` are the thin torch wrappers (lazy import).

trust_remote_code: the model's processing_/modeling_ modules import torchvision,
requests, peft, cv2, lmdb and decord at module top level (see the [models] extra);
from_pretrained fails without them even though we never touch the video paths.
"""

from __future__ import annotations

import io
import re

from ..contract import InferResponse, WireDetection

# Official model-card format: 4 normalized-int coords wrapped in <box> tokens.
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
# Detection prompt template (LocateAnythingWorker.detect); one category per call.
_PROMPT = "Locate all the instances that matches the following description: {query}."


def _parse_boxes(answer: str) -> list[tuple[int, int, int, int]]:
    """Pull every `<box>`-token quadruple (0-1000 normalized ints) out of the
    generated text, in order. Ignores any surrounding label/semantic tokens."""
    return [(int(a), int(b), int(c), int(d)) for a, b, c, d in _BOX_RE.findall(answer)]


def _to_wire(
    boxes: list[tuple[int, int, int, int]], label: str, w: int, h: int
) -> list[WireDetection]:
    """0-1000 normalized boxes for one category -> abs-px WireDetections.
    Scales by image dims, clamps to the frame, drops degenerate/inverted boxes.
    No score (generative model gives none)."""
    out: list[WireDetection] = []
    for x1, y1, x2, y2 in boxes:
        ax1, ay1 = x1 / 1000 * w, y1 / 1000 * h
        ax2, ay2 = x2 / 1000 * w, y2 / 1000 * h
        # tolerate a flipped corner ordering before clamping
        cx1, cx2 = sorted((ax1, ax2))
        cy1, cy2 = sorted((ay1, ay2))
        cx1, cy1 = max(0.0, cx1), max(0.0, cy1)
        cx2, cy2 = min(float(w), cx2), min(float(h), cy2)
        if cx2 <= cx1 or cy2 <= cy1:
            continue  # nothing left after clamping -> drop
        out.append(WireDetection(bbox=[cx1, cy1, cx2, cy2], label=label, score=None))
    return out


def _patch_remote_attn_signature(hf_model: str) -> None:  # pragma: no cover - needs transformers + remote code
    """transformers-5.x compatibility shim for LocateAnything's trust_remote_code.

    The model's remote `_check_and_adjust_attn_implementation` overrides were written
    against pre-5.0 transformers; 5.x's PreTrainedModel.__init__ now calls it with a
    new `allow_all_kernels` kwarg the overrides don't accept, so instantiation dies
    with `TypeError: ... unexpected keyword argument 'allow_all_kernels'`. The model
    bundles SEVERAL such classes (LocateAnything + its own vendored Qwen2 + MoonViT),
    so we resolve the dynamic class (which imports the whole remote package) and then
    wrap the override on EVERY class in the loaded `transformers_modules` that has the
    old signature. Narrow and reversible: touches only this model's dynamically-loaded
    classes, leaving OWLv2 (which relies on stock 5.x) on the unmodified path.
    """
    import inspect
    import sys

    from transformers.modeling_utils import PreTrainedModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # No-op unless the stock base actually passes `allow_all_kernels` (transformers
    # 5.x). We pin transformers==4.57.1 (LocateAnything's tested version), where this
    # is absent, so this stays dormant — it's a guard for a future transformers bump.
    base_sig = inspect.signature(PreTrainedModel._check_and_adjust_attn_implementation)
    if "allow_all_kernels" not in base_sig.parameters:
        return

    # Import the whole remote package (pulls in modeling_qwen2, modeling_vit, ...).
    get_class_from_dynamic_module(
        "modeling_locateanything.LocateAnythingForConditionalGeneration", hf_model
    )
    name = "_check_and_adjust_attn_implementation"

    def _needs_patch(fn) -> bool:
        params = inspect.signature(fn).parameters
        return "allow_all_kernels" not in params and not any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    for mod_name, module in list(sys.modules.items()):
        if "transformers_modules" not in mod_name or module is None:
            continue
        for klass in vars(module).values():
            if not inspect.isclass(klass):
                continue
            fn = klass.__dict__.get(name)
            if fn is None or not _needs_patch(fn):
                continue
            def _compat(self, *a, _orig=fn, **kw):
                kw.pop("allow_all_kernels", None)
                return _orig(self, *a, **kw)
            setattr(klass, name, _compat)


class LocateAnythingAdapter:
    """nvidia/LocateAnything-3B via transformers (trust_remote_code). torch + the
    custom remote modules import lazily in load()/detect()."""

    def __init__(self, hf_model: str) -> None:
        self.hf_model = hf_model
        self.ready = False
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._device = "cpu"
        self._dtype = None

    def load(self) -> None:  # pragma: no cover - needs torch + weights (GPU pod)
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16
        _patch_remote_attn_signature(self.hf_model)  # transformers-5.x compat (see below)
        self._tokenizer = AutoTokenizer.from_pretrained(self.hf_model, trust_remote_code=True)
        self._processor = AutoProcessor.from_pretrained(self.hf_model, trust_remote_code=True)
        self._model = (
            AutoModel.from_pretrained(self.hf_model, dtype=self._dtype, trust_remote_code=True)
            .to(self._device)
            .eval()
        )
        self.ready = True

    def _generate(self, image, query: str, params: dict) -> str:  # pragma: no cover - needs torch
        """One detection pass for a single category; returns the raw answer text.
        Mirrors LocateAnythingWorker.predict (the model-card serving reference)."""
        import torch

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _PROMPT.format(query=query)},
            ]}
        ]
        text = self._processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self._processor.process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            response = self._model.generate(
                pixel_values=inputs["pixel_values"].to(self._dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self._tokenizer,
                max_new_tokens=int(params.get("max_new_tokens", 2048)),
                use_cache=True,
                generation_mode=params.get("generation_mode", "hybrid"),
                temperature=float(params.get("temperature", 0.7)),
                do_sample=bool(params.get("do_sample", True)),
                top_p=float(params.get("top_p", 0.9)),
                repetition_penalty=float(params.get("repetition_penalty", 1.1)),
                verbose=False,
            )
        # generate returns the decoded answer string, or a (answer, history, stats) tuple
        return response[0] if isinstance(response, tuple) else response

    def detect(self, image_url: str, queries: list[str], params: dict) -> InferResponse:  # pragma: no cover - needs torch
        import httpx
        from PIL import Image

        resp = httpx.get(image_url, timeout=60.0)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        w, h = image.size

        dets: list[WireDetection] = []
        for q in queries or ["object"]:
            answer = self._generate(image, q, params)
            dets.extend(_to_wire(_parse_boxes(answer), q, w, h))
        return InferResponse(width=w, height=h, detections=dets)
