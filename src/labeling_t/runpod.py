"""RunPod provisioning — stand a model's vLLM endpoint up/down.

Part of the framework (not a loose script): the serving recipe lives with the
code so spinning a model up is one command, and the logic is importable/testable.

CLI (installed as `labeling-t-runpod`):
    labeling-t-runpod up       # rent GPU, serve, write endpoint -> .env
    labeling-t-runpod status   # balance + running pods
    labeling-t-runpod down     # delete this project's pods (stop billing)
    labeling-t-runpod gpus     # list GPU presets

Hardware comes from a PodSpec (gpu.py), the model from a ModelSpec (models.py).

Auth: RUNPOD_API_KEY from .env if set, else runpodctl's stored login.
WARNING: `up` rents a GPU and COSTS MONEY. `up` always sets an auto-terminate
backstop so nothing bills overnight.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from .config import load_env
from .gpu import DEFAULT_GPU, GPUS, get_pod
from .models import ModelSpec, get_spec

IMAGE = "vllm/vllm-openai:latest"   # GPU/disk/cloud/CUDA come from the PodSpec
NAME_PREFIX = "labeling-t"


def _env() -> dict:
    """Subprocess env with RUNPOD_API_KEY from .env (else runpodctl's own login)."""
    load_env()
    e = dict(os.environ)
    key = e.get("RUNPOD_API_KEY", "").strip()
    if key:
        e["RUNPOD_API_KEY"] = key
    else:
        e.pop("RUNPOD_API_KEY", None)
    return e


def _runpodctl(args: list[str], env: dict) -> str:
    r = subprocess.run(["runpodctl", *args], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"runpodctl {' '.join(args)} failed:\n{r.stderr or r.stdout}")
    return r.stdout


def _pods(env: dict) -> list[dict]:
    data = json.loads(_runpodctl(["pod", "list", "-o", "json"], env) or "[]")
    if isinstance(data, dict):
        data = data.get("pods") or data.get("data") or []
    return data or []


def _docker_args(spec: ModelSpec) -> str:
    if not spec.hf_model:
        raise SystemExit(f"spec {spec.key!r} has no hf_model — can't serve it")
    return (
        f"{spec.hf_model} --served-model-name {spec.name} "
        f"--host 0.0.0.0 --port 8000 {spec.serve_args}"
    ).strip()


def _proxy(pod_id: str) -> str:
    return f"https://{pod_id}-8000.proxy.runpod.net"


def _write_env_endpoint(prefix: str, url: str, env_path: str | Path = ".env") -> None:
    path = Path(env_path)
    var = f"{prefix}_ENDPOINT"
    lines = path.read_text().splitlines() if path.exists() else []
    for i, ln in enumerate(lines):
        if ln.strip().startswith(var + "="):
            lines[i] = f"{var}={url}"
            break
    else:
        lines.append(f"{var}={url}")
    path.write_text("\n".join(lines) + "\n")


def cmd_up(a, env) -> int:
    spec = get_spec(a.model)
    hw = get_pod(a.gpu)
    disk = a.disk or hw.disk_gb
    cloud = a.cloud or hw.cloud
    min_cuda = a.min_cuda or hw.min_cuda
    term = (datetime.now(timezone.utc) + timedelta(hours=a.hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    name = f"{NAME_PREFIX}-{spec.key.replace('_', '-')}"
    print(f"renting {hw.gpu_id} ({hw.vram_gb or '?'}GB, {cloud}) for {spec.name}")
    out = _runpodctl([
        "pod", "create", "--name", name,
        "--gpu-id", hw.gpu_id, "--gpu-count", str(hw.gpu_count),
        "--image", IMAGE, "--container-disk-in-gb", str(disk),
        "--ports", "8000/http", "--cloud-type", cloud,
        "--min-cuda-version", min_cuda,
        "--terminate-after", term,
        "--docker-args", _docker_args(spec),
        "-o", "json",
    ], env)
    pod = json.loads(out)
    pid = pod["id"]
    url = _proxy(pid)
    print(f"created pod {pid}  (${pod.get('costPerHr')}/hr, auto-terminate {term})")
    _write_env_endpoint(spec.env_prefix, url)
    print(f"endpoint -> {url}  (wrote {spec.env_prefix}_ENDPOINT to .env)")

    if a.no_wait:
        print(f"not waiting (--no-wait). Check: labeling-t-runpod ... or spike --model {spec.key} --check")
        return 0
    print(f"waiting for vLLM to serve (download + load, up to {a.timeout // 60} min)...")
    deadline = time.time() + a.timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/v1/models", timeout=10)
            if r.status_code == 200:
                print(f"READY — serving {[m['id'] for m in r.json().get('data', [])]}")
                return 0
        except httpx.HTTPError:
            pass
        time.sleep(15)
    print("timed out waiting. Pod is up — check RunPod GUI logs for progress/errors.", file=sys.stderr)
    return 1


def cmd_down(a, env) -> int:
    pods = _pods(env)
    targets = pods if a.all else [p for p in pods if str(p.get("name", "")).startswith(NAME_PREFIX)]
    if not targets:
        print("no matching pods running")
        return 0
    for p in targets:
        _runpodctl(["pod", "delete", p["id"]], env)
        print(f"deleted {p['id']} ({p.get('name')})")
    return 0


def cmd_status(a, env) -> int:
    bal = json.loads(_runpodctl(["user", "-o", "json"], env))
    print(f"balance: ${bal.get('clientBalance', 0):.2f} | spend/hr: ${bal.get('currentSpendPerHr', 0)}")
    pods = _pods(env)
    if not pods:
        print("no pods running")
        return 0
    for p in pods:
        print(f"  {p['id']}  {p.get('name')}  ${p.get('costPerHr')}/hr  {p.get('desiredStatus')}")
    return 0


def cmd_gpus(a, env) -> int:
    print("GPU presets (use as --gpu <key>, or pass any raw RunPod gpu-id):")
    for p in GPUS.values():
        print(f"  {p.key:9s} {p.vram_gb:>3}GB  {p.cloud:<9} {p.gpu_id:<28} {p.note}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="labeling-t-runpod",
                                description="RunPod provisioning for a model's vLLM endpoint")
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("up", help="rent a GPU and serve the model")
    up.add_argument("--model", default="qwen3_vl", help="model spec key (models.py)")
    up.add_argument("--gpu", default=DEFAULT_GPU,
                    help="GPU preset (rtx4090, rtx5090, a40, a100, h100...) or a raw RunPod gpu-id")
    up.add_argument("--disk", type=int, default=0, help="override preset container disk GB")
    up.add_argument("--cloud", default=None, help="override preset cloud (SECURE/COMMUNITY)")
    up.add_argument("--min-cuda", default=None, help="override preset min CUDA version")
    up.add_argument("--hours", type=float, default=3.0, help="auto-terminate after N hours")
    up.add_argument("--timeout", type=int, default=900, help="readiness wait, seconds")
    up.add_argument("--no-wait", action="store_true")
    up.set_defaults(func=cmd_up)

    dn = sub.add_parser("down", help="delete this project's pods (stop billing)")
    dn.add_argument("--all", action="store_true", help="delete ALL pods, not just labeling-t-*")
    dn.set_defaults(func=cmd_down)

    sub.add_parser("status", help="balance + running pods").set_defaults(func=cmd_status)
    sub.add_parser("gpus", help="list GPU presets").set_defaults(func=cmd_gpus)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a, _env())


if __name__ == "__main__":
    raise SystemExit(main())
