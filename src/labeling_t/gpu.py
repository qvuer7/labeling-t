"""Rentable GPU / pod presets for RunPod provisioning.

Separation of concerns:
  ModelSpec (models.py)  = WHAT to serve   (hf repo, vllm args, prompt, parser)
  PodSpec   (here)       = WHERE to serve it (GPU, VRAM, cloud, disk, CUDA)

A PodSpec is everything `runpodctl pod create` needs about the hardware. Add a
GPU by adding one entry to GPUS; or pass any raw RunPod gpu-id ad-hoc (see
scripts/runpod.py). gpu_id strings come from `runpodctl gpu list`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PodSpec:
    key: str                 # short selector, e.g. "rtx4090"
    gpu_id: str              # runpodctl --gpu-id (exact, from `runpodctl gpu list`)
    vram_gb: int
    min_cuda: str = "13.0"   # stock vllm-openai:latest needs CUDA 13 on the node
    cloud: str = "SECURE"    # SECURE | COMMUNITY
    disk_gb: int = 60
    gpu_count: int = 1
    note: str = ""           # informal guidance (approx fit / cost)


# Confirmed gpu_id strings (from `runpodctl gpu list`). Approx prices are
# guidance only — check live with `runpodctl gpu list` / the RunPod console.
RTX3090 = PodSpec("rtx3090", "NVIDIA GeForce RTX 3090", 24, note="cheapest 24GB (Ampere); <=8B VLM")
RTX4090 = PodSpec("rtx4090", "NVIDIA GeForce RTX 4090", 24, note="~$0.69/hr secure; sweet spot for <=8B")
RTX5090 = PodSpec("rtx5090", "NVIDIA GeForce RTX 5090", 32, note="newer/Blackwell; more headroom than 4090")
A40 = PodSpec("a40", "NVIDIA A40", 48, note="48GB; room for ~30B-A3B or large batches")
A100 = PodSpec("a100", "NVIDIA A100 80GB PCIe", 80, note="80GB; 30B-A3B / 32B class models")
H100 = PodSpec("h100", "NVIDIA H100 80GB HBM3", 80, note="fastest, priciest; large models / max throughput")

GPUS: dict[str, PodSpec] = {p.key: p for p in (RTX3090, RTX4090, RTX5090, A40, A100, H100)}

DEFAULT_GPU = "rtx4090"


def get_pod(key: str) -> PodSpec:
    """Look up a preset by key, or treat `key` as a raw RunPod gpu-id (ad-hoc,
    default pod settings) so any GPU works without a preset."""
    if key in GPUS:
        return GPUS[key]
    # Ad-hoc: caller passed a literal RunPod gpu-id like "NVIDIA L40S".
    return PodSpec(key=key, gpu_id=key, vram_gb=0, note="ad-hoc (no preset)")
