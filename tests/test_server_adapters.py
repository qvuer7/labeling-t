"""Server adapters: OWLv2 unpad/normalization (no torch), registry, runpod wiring."""

import pytest

from labeling_t.models import get_spec
from labeling_t.runpod import IMAGE, MODELS_IMAGE, _dcs_for_gpu, _serving
from labeling_t.server.adapters import (
    LocateAnythingAdapter,
    Owlv2Adapter,
    Sam2Adapter,
    StubAdapter,
    get_adapter,
)
from labeling_t.server.adapters.locateanything import _parse_boxes, _to_wire
from labeling_t.server.adapters.owlv2 import _finalize
from labeling_t.server.adapters.sam2 import _input_boxes


def test_finalize_unpads_padding_region_and_clamps():
    # 200x100 original -> OWLv2 pads to a 200x200 square; post_process boxes are in
    # padded-square pixels, labels already resolved to strings. In-frame box
    # kept+clamped; bottom-padding box dropped.
    boxes = [
        [10, 20, 110, 90],    # inside real frame -> keep as-is
        [10, 150, 60, 190],   # y1=150 in bottom padding (real h=100) -> DROP
        [180, 10, 260, 80],   # x2=260 > 200 -> clamp to 200
    ]
    dets = _finalize(boxes, [0.9, 0.8, 0.7], ["player", "ball", "player"],
                     orig_w=200, orig_h=100, threshold=0.5)
    assert [d.label for d in dets] == ["player", "player"]
    assert dets[0].bbox == [10.0, 20.0, 110.0, 90.0]
    assert dets[1].bbox == [180.0, 10.0, 200.0, 80.0]  # clamped x2


def test_finalize_drops_low_score_and_unresolved_label():
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    # score 0.2 < 0.5 -> drop; label None (couldn't resolve) -> drop
    assert _finalize(boxes, [0.2, 0.9], ["a", None], 100, 100, threshold=0.5) == []


def test_finalize_drops_degenerate_after_clamp():
    # zero-width box (x1==x2) inside the frame -> dropped by the cx2<=cx1 guard
    assert _finalize([[150, 50, 150, 90]], [0.9], ["a"], 200, 100, 0.1) == []


def test_registry_returns_unloaded_adapters_without_torch():
    assert isinstance(get_adapter("stub"), StubAdapter)
    a = get_adapter("owlv2")
    assert isinstance(a, Owlv2Adapter)
    assert a.ready is False  # constructed, not loaded -> no torch imported
    assert "owlv2" in a.hf_model
    with pytest.raises(KeyError):
        get_adapter("nope")


def test_registry_returns_locate_anything_unloaded():
    a = get_adapter("locate_anything")
    assert isinstance(a, LocateAnythingAdapter)
    assert a.ready is False  # constructed, not loaded -> no torch/remote-code imported
    assert a.hf_model == "nvidia/LocateAnything-3B"


def test_locateanything_parse_boxes_extracts_quadruples_in_order():
    answer = "player<box><10><20><300><400></box> ball <box><500><600><700><800></box>"
    assert _parse_boxes(answer) == [(10, 20, 300, 400), (500, 600, 700, 800)]
    assert _parse_boxes("no boxes here") == []


def test_locateanything_to_wire_scales_clamps_and_tags_label():
    # 0-1000 normalized -> abs px on a 200x100 frame; out-of-range box clamps,
    # flipped corners are reordered, label comes from the query (no score).
    boxes = [(0, 0, 500, 500), (250, 0, 1200, 500)]  # 2nd x2=1200/1000*200=240 -> clamp to 200
    dets = _to_wire(boxes, "player", w=200, h=100)
    assert [d.label for d in dets] == ["player", "player"]
    assert all(d.score is None for d in dets)
    assert dets[0].bbox == [0.0, 0.0, 100.0, 50.0]
    assert dets[1].bbox == [50.0, 0.0, 200.0, 50.0]


def test_locateanything_to_wire_drops_degenerate():
    # zero-area box (x1==x2) -> dropped
    assert _to_wire([(500, 100, 500, 900)], "ball", 200, 100) == []


def test_registry_returns_sam2_unloaded():
    a = get_adapter("sam2")
    assert isinstance(a, Sam2Adapter)
    assert a.ready is False  # constructed, not loaded -> no torch imported
    assert "sam2" in a.hf_model


def test_sam2_input_boxes_wraps_to_nested_prompt_shape():
    # SAM2 wants [batch][num_boxes][4]; one image per call -> single batch entry.
    out = _input_boxes([[10, 20, 30, 40], [1, 2, 3, 4]])
    assert out == [[[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]]]
    assert _input_boxes([]) == [[]]


def test_stub_adapter_detect_shape():
    r = get_adapter("stub").detect("http://x", ["player", "ball"], {})
    assert (r.width, r.height) == (640, 480)
    assert [d.label for d in r.detections] == ["player", "ball"]


def test_serving_recipe_selects_backend():
    v = _serving(get_spec("qwen3_vl"))   # vllm
    assert v["image"] == IMAGE and v["health"] == "/v1/models" and v["env"] == {}
    assert v["docker_args"]               # vLLM passes docker args

    t = _serving(get_spec("owlv2"))       # transformers
    assert t["image"] == MODELS_IMAGE and t["health"] == "/health"
    assert t["docker_args"] == ""
    assert t["env"]["MODEL"] == "owlv2"   # runpodctl gets ONE json object, not KEY=VAL flags
    assert "HF_MODEL" in t["env"]

    la = _serving(get_spec("locate_anything"))   # also transformers backend
    assert la["image"] == MODELS_IMAGE and la["health"] == "/health"
    assert la["env"]["MODEL"] == "locate_anything"
    assert la["env"]["HF_MODEL"] == "nvidia/LocateAnything-3B"


def test_dcs_for_gpu_filters_by_stock_and_ranks_best_first():
    # shape mirrors `runpodctl datacenter list -o json`
    datacenters = [
        {"id": "EU-RO-1", "gpuAvailability": [
            {"gpuId": "NVIDIA GeForce RTX 5090", "stockStatus": "Low"},
            {"gpuId": "NVIDIA A40", "stockStatus": "High"}]},
        {"id": "US-NC-1", "gpuAvailability": [
            {"gpuId": "NVIDIA GeForce RTX 5090", "stockStatus": ""}]},   # listed but empty -> drop
        {"id": "EU-CZ-1", "gpuAvailability": [
            {"gpuId": "NVIDIA GeForce RTX 5090", "stockStatus": "High"}]},
        {"id": "AP-IN-1", "gpuAvailability": [
            {"gpuId": "NVIDIA H100 80GB HBM3", "stockStatus": "High"}]},  # different gpu -> ignore
    ]
    out = _dcs_for_gpu(datacenters, "NVIDIA GeForce RTX 5090")
    # only DCs with non-empty 5090 stock, High before Low
    assert out == [("EU-CZ-1", "High"), ("EU-RO-1", "Low")]
    assert _dcs_for_gpu(datacenters, "NVIDIA A40") == [("EU-RO-1", "High")]
    assert _dcs_for_gpu([], "anything") == []


# ---- --json envelope on the runpod CLI (runpodctl stubbed) ----------------

def test_status_json_envelope(monkeypatch, capsys):
    import json

    import labeling_t.runpod as rp

    def fake_runpodctl(args, env):
        if args[0] == "user":
            return json.dumps({"clientBalance": 12.5, "currentSpendPerHr": 0.69})
        if args[0] == "pod":
            return json.dumps([{"id": "abc123", "name": "labeling-t-sam2",
                                "costPerHr": 0.69, "desiredStatus": "RUNNING"}])
        raise AssertionError(args)

    monkeypatch.setattr(rp, "_runpodctl", fake_runpodctl)
    rc = rp.main(["status", "--json"])
    out, err = capsys.readouterr()
    envelope = json.loads(out)  # stdout is EXACTLY one JSON envelope
    assert rc == 0 and envelope["ok"] is True
    assert envelope["result"]["balance"] == 12.5
    assert envelope["result"]["pods"][0]["id"] == "abc123"
    assert "balance: $12.50" in err  # prose demoted to stderr


def test_status_without_json_is_prose(monkeypatch, capsys):
    import json

    import labeling_t.runpod as rp

    monkeypatch.setattr(rp, "_runpodctl", lambda args, env: json.dumps(
        {"clientBalance": 1.0, "currentSpendPerHr": 0} if args[0] == "user" else []))
    rc = rp.main(["status"])
    out, err = capsys.readouterr()
    assert rc == 0 and "no pods running" in out and not err


def test_gpus_json_envelope(capsys):
    import json

    import labeling_t.runpod as rp

    rc = rp.main(["gpus", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["ok"] is True
    keys = {g["key"] for g in envelope["result"]["gpus"]}
    assert "a40" in keys and "rtx4090" in keys
    assert envelope["result"]["default"] == "rtx4090"


def test_datacenters_json_error_when_no_stock(monkeypatch, capsys):
    import json

    import labeling_t.runpod as rp

    monkeypatch.setattr(rp, "_datacenters_with_stock", lambda gpu_id, env: [])
    rc = rp.main(["datacenters", "--gpu", "a40", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 1 and envelope["ok"] is False
    assert envelope["error"]["gpu_id"] == "NVIDIA A40"  # ok mirrors the exit code


def test_runpodctl_failure_becomes_error_envelope(monkeypatch, capsys):
    import json

    import labeling_t.runpod as rp

    def boom(args, env):
        raise SystemExit("runpodctl user failed:\nboom")

    monkeypatch.setattr(rp, "_runpodctl", boom)
    rc = rp.main(["status", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 1 and envelope["ok"] is False and "boom" in envelope["error"]["message"]
