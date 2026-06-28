"""Server adapters: OWLv2 unpad/normalization (no torch), registry, runpod wiring."""

import pytest

from labeling_t.models import get_spec
from labeling_t.runpod import IMAGE, MODELS_IMAGE, _serving
from labeling_t.server.adapters import Owlv2Adapter, StubAdapter, get_adapter
from labeling_t.server.adapters.owlv2 import _finalize


def test_finalize_unpads_padding_region_and_clamps():
    # 200x100 original -> OWLv2 pads to a 200x200 square; post_process boxes are in
    # padded-square pixels. In-frame box kept+clamped; bottom-padding box dropped.
    boxes = [
        [10, 20, 110, 90],    # inside real frame -> keep as-is
        [10, 150, 60, 190],   # y1=150 in bottom padding (real h=100) -> DROP
        [180, 10, 260, 80],   # x2=260 > 200 -> clamp to 200
    ]
    dets = _finalize(boxes, [0.9, 0.8, 0.7], [0, 1, 0], ["player", "ball"],
                     orig_w=200, orig_h=100, threshold=0.5)
    assert [d.label for d in dets] == ["player", "player"]
    assert dets[0].bbox == [10.0, 20.0, 110.0, 90.0]
    assert dets[1].bbox == [180.0, 10.0, 200.0, 80.0]  # clamped x2


def test_finalize_drops_low_score_and_bad_label():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    # score 0.2 < 0.5 -> drop; label id 9 out of range for 1-query list -> drop
    assert _finalize(boxes, [0.2, 0.9], [0, 9], ["a"], 100, 100, threshold=0.5) == []


def test_finalize_drops_degenerate_after_clamp():
    # box entirely at/over the right edge collapses to zero width -> dropped
    assert _finalize([[200, 10, 260, 80]], [0.9], [0], ["a"], 200, 100, 0.1) == []


def test_registry_returns_unloaded_adapters_without_torch():
    assert isinstance(get_adapter("stub"), StubAdapter)
    a = get_adapter("owlv2")
    assert isinstance(a, Owlv2Adapter)
    assert a.ready is False  # constructed, not loaded -> no torch imported
    assert "owlv2" in a.hf_model
    with pytest.raises(KeyError):
        get_adapter("nope")


def test_stub_adapter_detect_shape():
    r = get_adapter("stub").detect("http://x", ["player", "ball"], {})
    assert (r.width, r.height) == (640, 480)
    assert [d.label for d in r.detections] == ["player", "ball"]


def test_serving_recipe_selects_backend():
    v = _serving(get_spec("qwen3_vl"))   # vllm
    assert v["image"] == IMAGE and v["health"] == "/v1/models" and v["env"] == []
    assert v["docker_args"]               # vLLM passes docker args

    t = _serving(get_spec("owlv2"))       # transformers
    assert t["image"] == MODELS_IMAGE and t["health"] == "/health"
    assert t["docker_args"] == ""
    assert "MODEL=owlv2" in t["env"]
    assert any(e.startswith("HF_MODEL=") for e in t["env"])
