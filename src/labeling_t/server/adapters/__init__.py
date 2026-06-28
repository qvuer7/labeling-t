"""Adapter registry — mirror of the framework's ModelSpec registry, server-side.

`MODEL=<key>` (env) selects which adapter the pod loads. Importing this module is
torch-free: adapter classes only import torch inside load()/detect(), so the
registry can be built without a GPU. Adding a model = one entry here + one
adapter file.
"""

from __future__ import annotations

from .base import ModelAdapter, StubAdapter
from .owlv2 import Owlv2Adapter

# key -> factory(taking the HF model id). Kept as factories so nothing loads
# until the server picks ONE and calls load().
_FACTORIES = {
    "stub": lambda hf: StubAdapter(),
    "owlv2": lambda hf: Owlv2Adapter(hf or "google/owlv2-base-patch16-ensemble"),
    # PR-2: "grounding_dino", "locate_anything"
}


def get_adapter(model: str, hf_model: str | None = None) -> ModelAdapter:
    try:
        return _FACTORIES[model](hf_model)
    except KeyError:
        raise KeyError(f"unknown model {model!r}; known: {sorted(_FACTORIES)}") from None


__all__ = ["ModelAdapter", "StubAdapter", "Owlv2Adapter", "get_adapter"]
