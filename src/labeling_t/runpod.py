"""RunPod provisioning — stand a model's vLLM endpoint up/down.

Part of the framework (not a loose script): the serving recipe lives with the
code so spinning a model up is one command, and the logic is importable/testable.

CLI (installed as `labeling-t-runpod`):
    labeling-t-runpod up           # rent GPU, serve, write endpoint -> .env
    labeling-t-runpod status       # balance + running pods (shows pod ids)
    labeling-t-runpod down         # delete this project's pod (stop billing)
    labeling-t-runpod down <id>    # delete a specific pod (when several run)
    labeling-t-runpod down --all   # delete every labeling-t-* pod
    labeling-t-runpod gpus         # list GPU presets

With multiple instances up, `down` (no args) refuses to guess and lists the
running pods; target one by id or use --all.

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
from typing import Callable

import httpx

from .config import load_env
from .gpu import DEFAULT_GPU, GPUS, get_pod
from .models import ModelSpec, get_spec

IMAGE = "vllm/vllm-openai:latest"   # GPU/disk/cloud/CUDA come from the PodSpec
# Our transformers model-server image. PUBLIC GHCR on purpose -> RunPod needs no
# registry credentials to pull it (keeps PR-1 launch simple). Push it first:
#   docker build -t $MODELS_IMAGE . && docker push $MODELS_IMAGE
MODELS_IMAGE = "ghcr.io/qvuer7/labeling-t-models:latest"
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


def _serving(spec: ModelSpec) -> dict:
    """Per-backend serving recipe: image, docker-args, extra pod env, and the
    HTTP path that signals 'ready'. vLLM = its official image + /v1/models; our
    transformers server = the GHCR image + /health, with MODEL/HF_* via env."""
    if spec.backend == "transformers":
        env = {"MODEL": spec.key}
        if spec.hf_model:
            env["HF_MODEL"] = spec.hf_model
        token = os.environ.get("HF_TOKEN", "").strip()
        if token:  # gated weights (e.g. LocateAnything-3B) need an HF token in the pod
            env["HF_TOKEN"] = token
        return {"image": MODELS_IMAGE, "docker_args": "", "env": env, "health": "/health"}
    return {"image": IMAGE, "docker_args": _docker_args(spec), "env": {}, "health": "/v1/models"}


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


class AmbiguousPods(Exception):
    """More than one project pod is running and no target was given — refuse to
    guess which to delete. Carries the candidate pods so callers can list them."""

    def __init__(self, pods: list[dict]):
        self.pods = pods
        super().__init__(f"{len(pods)} {NAME_PREFIX} pods running; specify id(s) or all")


def start_pod(
    model: str = "qwen3_vl",
    *,
    gpu: str = DEFAULT_GPU,
    hours: float = 3.0,
    disk: int = 0,
    cloud: str | None = None,
    min_cuda: str | None = None,
    timeout: int = 900,
    wait: bool = True,
    env: dict | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Rent a GPU, serve `model` on vLLM, write its endpoint to .env. Returns a
    dict {id, endpoint, cost_per_hr, terminate_after, ready, served}. `log` is a
    progress sink (print for the CLI, a Job's log for the web UI). With wait=True
    it polls /v1/models until the model serves or `timeout` seconds elapse."""
    env = env or _env()
    spec = get_spec(model)
    hw = get_pod(gpu)
    disk = disk or hw.disk_gb
    cloud = cloud or hw.cloud
    min_cuda = min_cuda or hw.min_cuda
    term = (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    name = f"{NAME_PREFIX}-{spec.key.replace('_', '-')}"
    serving = _serving(spec)
    log(f"renting {hw.gpu_id} ({hw.vram_gb or '?'}GB, {cloud}) for {spec.name} [{spec.backend}]")
    cmd = [
        "pod", "create", "--name", name,
        "--gpu-id", hw.gpu_id, "--gpu-count", str(hw.gpu_count),
        "--image", serving["image"], "--container-disk-in-gb", str(disk),
        "--ports", "8000/http", "--cloud-type", cloud,
        "--min-cuda-version", min_cuda,
        "--terminate-after", term,
    ]
    if serving["docker_args"]:
        cmd += ["--docker-args", serving["docker_args"]]
    if serving["env"]:               # runpodctl wants env as ONE json object string
        cmd += ["--env", json.dumps(serving["env"])]
    cmd += ["-o", "json"]
    out = _runpodctl(cmd, env)
    pod = json.loads(out)
    pid = pod["id"]
    url = _proxy(pid)
    log(f"created pod {pid}  (${pod.get('costPerHr')}/hr, auto-terminate {term})")
    _write_env_endpoint(spec.env_prefix, url)
    log(f"endpoint -> {url}  (wrote {spec.env_prefix}_ENDPOINT to .env)")
    info = {"id": pid, "endpoint": url, "cost_per_hr": pod.get("costPerHr"),
            "terminate_after": term, "model": spec.key, "ready": False, "served": []}
    if not wait:
        return info
    health = serving["health"]
    log(f"waiting for the model to be ready (download + load, up to {timeout // 60} min)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}{health}", timeout=10)
            if r.status_code == 200:
                if health == "/v1/models":  # vLLM: served list == ready
                    info["served"] = [m["id"] for m in r.json().get("data", [])]
                    info["ready"] = True
                elif r.json().get("ready"):  # our /health: ready only after weights load
                    info["served"] = [r.json().get("model")]
                    info["ready"] = True
                if info["ready"]:
                    log(f"READY — serving {info['served']}")
                    return info
        except httpx.HTTPError:
            pass
        time.sleep(15)
    log("timed out waiting. Pod is up — check RunPod GUI logs for progress/errors.")
    return info


def stop_pods(env: dict | None = None, *, pods: list[str] | None = None, all: bool = False) -> list[dict]:
    """Delete pods (stop billing). With explicit `pods` ids, deletes exactly
    those; with all=True, every project pod; otherwise the single project pod.
    Raises AmbiguousPods when several project pods run and no target was given,
    ValueError for unknown ids. Returns the deleted pods [{id, name}]."""
    env = env or _env()
    running = _pods(env)
    ours = [p for p in running if str(p.get("name", "")).startswith(NAME_PREFIX)]
    if pods:  # explicit id(s) -> delete exactly those (any pod, ours or not)
        want = set(pods)
        targets = [p for p in running if p["id"] in want]
        missing = want - {p["id"] for p in targets}
        if missing:
            raise ValueError(f"no such pod(s): {', '.join(sorted(missing))}")
    elif all:  # every pod of THIS project (stop all our billing), not unrelated pods
        targets = ours
    elif len(ours) > 1:  # ambiguous: don't nuke a sibling instance by accident
        raise AmbiguousPods(ours)
    else:  # zero or one of ours -> the common single-instance case
        targets = ours
    for p in targets:
        _runpodctl(["pod", "delete", p["id"]], env)
    return [{"id": p["id"], "name": p.get("name")} for p in targets]


def list_pods_with_balance(env: dict | None = None) -> dict:
    """Account balance + running pods as structured data. Returns
    {balance, spend_per_hr, pods: [{id, name, cost_per_hr, status}]}."""
    env = env or _env()
    bal = json.loads(_runpodctl(["user", "-o", "json"], env))
    return {
        "balance": bal.get("clientBalance", 0),
        "spend_per_hr": bal.get("currentSpendPerHr", 0),
        "pods": [
            {"id": p["id"], "name": p.get("name"),
             "cost_per_hr": p.get("costPerHr"), "status": p.get("desiredStatus")}
            for p in _pods(env)
        ],
    }


def cmd_up(a, env) -> int:
    info = start_pod(a.model, gpu=a.gpu, hours=a.hours, disk=a.disk, cloud=a.cloud,
                     min_cuda=a.min_cuda, timeout=a.timeout, wait=not a.no_wait, env=env)
    if a.no_wait:
        print(f"not waiting (--no-wait). Check: labeling-t-runpod status  (--model {a.model})")
        return 0
    return 0 if info["ready"] else 1


def cmd_down(a, env) -> int:
    try:
        deleted = stop_pods(env, pods=a.pods or None, all=a.all)
    except AmbiguousPods as exc:
        print(f"{len(exc.pods)} {NAME_PREFIX} pods running — pass a pod id, or --all for all of them:",
              file=sys.stderr)
        for p in exc.pods:
            print(f"  {p['id']}  {p.get('name')}  ${p.get('costPerHr')}/hr", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not deleted:
        print("no matching pods running")
        return 0
    for p in deleted:
        print(f"deleted {p['id']} ({p['name']})")
    return 0


def cmd_status(a, env) -> int:
    s = list_pods_with_balance(env)
    print(f"balance: ${s['balance']:.2f} | spend/hr: ${s['spend_per_hr']}")
    if not s["pods"]:
        print("no pods running")
        return 0
    for p in s["pods"]:
        print(f"  {p['id']}  {p['name']}  ${p['cost_per_hr']}/hr  {p['status']}")
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

    dn = sub.add_parser("down", help="delete a pod by id, or this project's pod (stop billing)")
    dn.add_argument("pods", nargs="*", metavar="POD_ID",
                    help="specific pod id(s) to delete (from `status`); omit to target this project's pod")
    dn.add_argument("--all", action="store_true",
                    help=f"delete ALL {NAME_PREFIX}-* pods (use when several are running)")
    dn.set_defaults(func=cmd_down)

    sub.add_parser("status", help="balance + running pods").set_defaults(func=cmd_status)
    sub.add_parser("gpus", help="list GPU presets").set_defaults(func=cmd_gpus)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a, _env())


if __name__ == "__main__":
    raise SystemExit(main())
