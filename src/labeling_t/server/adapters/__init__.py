"""Adapter registry — mirror of the framework's ModelSpec registry, server-side.

`MODEL=<key>` (env) selects which adapter the pod loads. Importing this module is
torch-free: adapter classes only import torch inside load()/detect(), so the
registry can be built without a GPU. Adding a model = one entry here + one
adapter file.
"""

from __future__ import annotations

from .base import ModelAdapter, StubAdapter
from .locateanything import LocateAnythingAdapter
from .owlv2 import Owlv2Adapter
from .sam2 import Sam2Adapter
from .vitpose import VitPoseAdapter

# key -> factory(taking the HF model id). Kept as factories so nothing loads
# until the server picks ONE and calls load().
_FACTORIES = {
    "stub": lambda hf: StubAdapter(),
    "owlv2": lambda hf: Owlv2Adapter(hf or "google/owlv2-base-patch16-ensemble"),
    "locate_anything": lambda hf: LocateAnythingAdapter(hf or "nvidia/LocateAnything-3B"),
    "sam2": lambda hf: Sam2Adapter(hf or "facebook/sam2.1-hiera-large"),
    "vitpose": lambda hf: VitPoseAdapter(hf or "usyd-community/vitpose-base-simple"),
    # PR-2: "grounding_dino"
}


def get_adapter(model: str, hf_model: str | None = None) -> ModelAdapter:
    try:
        return _FACTORIES[model](hf_model)
    except KeyError:
        raise KeyError(f"unknown model {model!r}; known: {sorted(_FACTORIES)}") from None


__all__ = [
    "ModelAdapter", "StubAdapter", "Owlv2Adapter", "LocateAnythingAdapter",
    "Sam2Adapter", "VitPoseAdapter", "get_adapter",
]
