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

Every subcommand takes --json: one machine-readable envelope on stdout, prose
(including `up` progress narration) to stderr. See output.py.

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
from .output import emit, fail, json_flag, note

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


# Stock labels in `runpodctl datacenter list`, best first. Pods placed by `pod
# create` without a datacenter hint get blind-placed onto one machine that often
# can't host the pod (esp. for scarce GPUs in few DCs) -> "no resources"/"no
# instances". Targeting the DCs that actually report stock is what the web console
# does for you; we replicate it so `up` finds the GPU the console can see.
_STOCK_RANK = {"High": 0, "Medium": 1, "Low": 2}


def _dcs_for_gpu(datacenters: list[dict], gpu_id: str) -> list[tuple[str, str]]:
    """Pure: from `runpodctl datacenter list` JSON, the (dc_id, stock) pairs that
    currently report non-empty stock for `gpu_id`, best-stock first. Pure so the
    selection logic is unit-tested without hitting RunPod."""
    out = [
        (dc["id"], ga["stockStatus"])
        for dc in datacenters
        for ga in dc.get("gpuAvailability", [])
        if ga.get("gpuId") == gpu_id and ga.get("stockStatus")
    ]
    return sorted(out, key=lambda ds: _STOCK_RANK.get(ds[1], 3))


def _datacenters_with_stock(gpu_id: str, env: dict) -> list[tuple[str, str]]:
    """(dc_id, stock) pairs that have `gpu_id` in stock now. [] if none or if the
    query fails (caller then deploys without a DC hint, the old behavior)."""
    try:
        data = json.loads(_runpodctl(["datacenter", "list", "-o", "json"], env) or "[]")
    except SystemExit:
        return []
    return _dcs_for_gpu(data if isinstance(data, list) else [], gpu_id)


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
    data_center: str | None = None,
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
    # Candidate datacenters to TRY IN ORDER. runpodctl honors only ONE
    # --data-center-id per create, so for a scarce GPU we try each stocked DC in
    # turn until one actually places (Low stock is racy; one DC out != all out).
    # Explicit --data-center wins; community cloud uses its own host pool (the DC
    # list is secure-oriented), so it gets no hint.
    if data_center:
        candidates: list[str | None] = [d.strip() for d in data_center.split(",") if d.strip()]
        log(f"datacenters: {', '.join(c for c in candidates if c)} (forced)")
    elif cloud.upper() == "SECURE":
        avail = _datacenters_with_stock(hw.gpu_id, env)
        candidates = [d for d, _ in avail] or [None]
        if avail:
            log(f"datacenters with {hw.gpu_id} in stock: " + ", ".join(f"{d}({s})" for d, s in avail))
        else:
            log(f"warning: no datacenter reports {hw.gpu_id} in stock right now — trying anyway")
    else:
        candidates = [None]
    base = [
        "pod", "create", "--name", name,
        "--gpu-id", hw.gpu_id, "--gpu-count", str(hw.gpu_count),
        "--image", serving["image"], "--container-disk-in-gb", str(disk),
        "--ports", "8000/http", "--cloud-type", cloud,
        "--min-cuda-version", min_cuda,
        "--terminate-after", term,
    ]
    if serving["docker_args"]:
        base += ["--docker-args", serving["docker_args"]]
    if serving["env"]:               # runpodctl wants env as ONE json object string
        base += ["--env", json.dumps(serving["env"])]
    base += ["-o", "json"]

    pod = None
    errors: list[str] = []
    for dc in candidates:
        cmd = base + (["--data-center-ids", dc] if dc else [])
        try:
            pod = json.loads(_runpodctl(cmd, env))
            if dc:
                log(f"placed in {dc}")
            break
        except SystemExit as exc:     # this DC can't place it — try the next
            errors.append(f"{dc or 'auto'}: {str(exc).strip().splitlines()[-1]}")
    if pod is None:
        raise SystemExit(
            f"could not place {hw.gpu_id} on any candidate datacenter "
            f"({cloud}). Tried:\n  " + "\n  ".join(errors)
            + "\n  (scarce GPU — retry, try --cloud COMMUNITY, or a different --gpu)"
        )
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
    # under --json, start_pod's progress narration must stay off stdout
    log = (lambda m: print(m, file=sys.stderr)) if a.json else print
    info = start_pod(a.model, gpu=a.gpu, hours=a.hours, disk=a.disk, cloud=a.cloud,
                     min_cuda=a.min_cuda, data_center=a.data_center, timeout=a.timeout,
                     wait=not a.no_wait, env=env, log=log)
    if a.no_wait:
        note(a, f"not waiting (--no-wait). Check: labeling-t-runpod status  (--model {a.model})")
        return emit(a, info)
    if not info["ready"]:
        return fail(a, "model did not become ready before the timeout "
                       "(pod is up and billing — check RunPod GUI logs)", result=info)
    return emit(a, info)


def cmd_down(a, env) -> int:
    try:
        deleted = stop_pods(env, pods=a.pods or None, all=a.all)
    except AmbiguousPods as exc:
        lines = [f"{len(exc.pods)} {NAME_PREFIX} pods running — pass a pod id, or --all for all of them:"]
        lines += [f"  {p['id']}  {p.get('name')}  ${p.get('costPerHr')}/hr" for p in exc.pods]
        return fail(a, "\n".join(lines),
                    pods=[{"id": p["id"], "name": p.get("name"), "cost_per_hr": p.get("costPerHr")}
                          for p in exc.pods])
    except ValueError as exc:
        return fail(a, str(exc))
    if not deleted:
        return emit(a, {"deleted": []}, "no matching pods running")
    return emit(a, {"deleted": deleted},
                "\n".join(f"deleted {p['id']} ({p['name']})" for p in deleted))


def cmd_status(a, env) -> int:
    s = list_pods_with_balance(env)
    lines = [f"balance: ${s['balance']:.2f} | spend/hr: ${s['spend_per_hr']}"]
    if not s["pods"]:
        lines.append("no pods running")
    else:
        lines += [f"  {p['id']}  {p['name']}  ${p['cost_per_hr']}/hr  {p['status']}" for p in s["pods"]]
    return emit(a, s, "\n".join(lines))


def cmd_gpus(a, env) -> int:
    from dataclasses import asdict

    lines = ["GPU presets (use as --gpu <key>, or pass any raw RunPod gpu-id):"]
    lines += [f"  {p.key:9s} {p.vram_gb:>3}GB  {p.cloud:<9} {p.gpu_id:<28} {p.note}"
              for p in GPUS.values()]
    return emit(a, {"gpus": [asdict(p) for p in GPUS.values()], "default": DEFAULT_GPU},
                "\n".join(lines))


def cmd_datacenters(a, env) -> int:
    gpu_id = get_pod(a.gpu).gpu_id
    avail = _datacenters_with_stock(gpu_id, env)
    if not avail:
        return fail(a, f"no datacenter reports {gpu_id} in stock right now", gpu_id=gpu_id)
    lines = [f"{gpu_id} in stock at (best first — `up` targets these automatically):"]
    lines += [f"  {dc_id:12s} {stock}" for dc_id, stock in avail]
    return emit(a, {"gpu_id": gpu_id,
                    "datacenters": [{"id": d, "stock": s} for d, s in avail]},
                "\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="labeling-t-runpod",
                                description="RunPod provisioning for a model's vLLM endpoint")
    sub = p.add_subparsers(dest="cmd", required=True)
    jf = [json_flag()]  # every subcommand takes --json (envelope on stdout)

    up = sub.add_parser("up", help="rent a GPU and serve the model", parents=jf)
    up.add_argument("--model", default="qwen3_vl", help="model spec key (models.py)")
    up.add_argument("--gpu", default=DEFAULT_GPU,
                    help="GPU preset (rtx4090, rtx5090, a40, a100, h100...) or a raw RunPod gpu-id")
    up.add_argument("--disk", type=int, default=0, help="override preset container disk GB")
    up.add_argument("--cloud", default=None, help="override preset cloud (SECURE/COMMUNITY)")
    up.add_argument("--min-cuda", default=None, help="override preset min CUDA version")
    up.add_argument("--data-center", default=None,
                    help="force datacenter(s), comma-separated (e.g. EU-RO-1). Default: auto-pick "
                         "DCs that report the GPU in stock (see `datacenters --gpu <preset>`)")
    up.add_argument("--hours", type=float, default=3.0, help="auto-terminate after N hours")
    up.add_argument("--timeout", type=int, default=900, help="readiness wait, seconds")
    up.add_argument("--no-wait", action="store_true")
    up.set_defaults(func=cmd_up)

    dn = sub.add_parser("down", help="delete a pod by id, or this project's pod (stop billing)", parents=jf)
    dn.add_argument("pods", nargs="*", metavar="POD_ID",
                    help="specific pod id(s) to delete (from `status`); omit to target this project's pod")
    dn.add_argument("--all", action="store_true",
                    help=f"delete ALL {NAME_PREFIX}-* pods (use when several are running)")
    dn.set_defaults(func=cmd_down)

    sub.add_parser("status", help="balance + running pods", parents=jf).set_defaults(func=cmd_status)
    sub.add_parser("gpus", help="list GPU presets", parents=jf).set_defaults(func=cmd_gpus)

    dc = sub.add_parser("datacenters", help="show which datacenters have a GPU in stock (what `up` auto-targets)",
                        parents=jf)
    dc.add_argument("--gpu", default=DEFAULT_GPU, help="GPU preset or a raw RunPod gpu-id")
    dc.set_defaults(func=cmd_datacenters)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.func(a, _env())
    except SystemExit as exc:
        # runpodctl / provisioning failures raise SystemExit with a message;
        # under --json that message must still arrive as an ok=false envelope.
        if getattr(a, "json", False) and isinstance(exc.code, str):
            return fail(a, exc.code)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
