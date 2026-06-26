"""GPU presets: registry lookup + ad-hoc passthrough."""

from labeling_t.gpu import DEFAULT_GPU, GPUS, PodSpec, get_pod


def test_default_gpu_is_registered():
    assert DEFAULT_GPU in GPUS


def test_preset_lookup_returns_hardware():
    p = get_pod("rtx4090")
    assert p.gpu_id == "NVIDIA GeForce RTX 4090"
    assert p.vram_gb == 24
    assert p.min_cuda == "13.0"


def test_bigger_preset_has_more_vram():
    assert get_pod("a100").vram_gb == 80
    assert get_pod("h100").vram_gb == 80


def test_unknown_key_becomes_adhoc_raw_gpu_id():
    # any literal RunPod gpu-id works without a preset
    p = get_pod("NVIDIA L40S")
    assert isinstance(p, PodSpec)
    assert p.gpu_id == "NVIDIA L40S"
    assert p.vram_gb == 0
    assert "ad-hoc" in p.note


def test_presets_have_distinct_gpu_ids():
    ids = [p.gpu_id for p in GPUS.values()]
    assert len(ids) == len(set(ids))
