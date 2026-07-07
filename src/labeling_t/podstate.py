"""Runtime pod state — which endpoints are up right now, on disk, not in .env.

`labeling-t-runpod up` records every pod it creates in `.labeling-t/pods.json`
(cwd-relative, gitignored, atomic writes); inference commands resolve their
endpoint from it. `.env` stays a SECRETS file — nothing in the framework writes
it. The file is a cache of what WE started, not the truth about what is running:
`labeling-t-runpod status` reconciles it against the live RunPod list (prunes
dead ids, adopts labeling-t-* pods created elsewhere), and a dead-but-recorded
pod simply fails to connect — recovery is `status --json`.

File shape (version 1), a dict keyed by pod id so upsert/remove are natural:

    {"version": 1, "pods": {"<id>": {id, model, env_prefix, endpoint, gpu,
                                     cost_per_hr, created_at, terminate_after,
                                     ready}}}

Endpoint resolution precedence (resolve_endpoint):
    explicit --endpoint flag
    > newest unexpired pods.json entry for the model
    > {PREFIX}_ENDPOINT env var (deprecated; stderr note)
    > the spec's baked-in default (SaaS providers)

Stdlib-only on purpose: state must be readable/writable from anywhere without
dragging in httpx/boto3. Tests point LABELING_T_PODS at a tmp file.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

VERSION = 1
NAME_PREFIX = "labeling-t"


def state_path() -> Path:
    """Where pods.json lives: $LABELING_T_PODS override (tests), else cwd."""
    return Path(os.environ.get("LABELING_T_PODS") or ".labeling-t/pods.json")


def load_pods() -> dict[str, dict]:
    """The recorded pods, keyed by id. Missing or corrupt file -> {} (the state
    is a cache; a broken cache must never block an inference command)."""
    path = state_path()
    try:
        data = json.loads(path.read_text())
        pods = data.get("pods", {})
        return pods if isinstance(pods, dict) else {}
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError, AttributeError):
        print(f"warning: unreadable pod state {path} — treating as empty", file=sys.stderr)
        return {}


def _save(pods: dict[str, dict]) -> None:
    """Atomic rewrite (tmp + os.replace) so a crash never leaves half a file."""
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"version": VERSION, "pods": pods}, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def record_pod(entry: dict) -> None:
    """Upsert one pod entry (keyed by entry['id'])."""
    pods = load_pods()
    pods[entry["id"]] = entry
    _save(pods)


def remove_pods(ids: Iterable[str]) -> list[str]:
    """Drop entries by id; unknown ids are ignored. Returns the ids removed."""
    pods = load_pods()
    removed = [i for i in ids if pods.pop(i, None) is not None]
    if removed:
        _save(pods)
    return removed


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _expired(entry: dict) -> bool:
    """True when the pod's auto-terminate time has passed. Missing/unparseable
    terminate_after counts as unexpired (adopted pods don't carry one)."""
    term = entry.get("terminate_after")
    if not term:
        return False
    try:
        dt = datetime.strptime(term, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return dt <= _now()


def reconcile(live: list[dict]) -> dict:
    """Sync state with the live pod list (from `runpodctl pod list`): the live
    list is the truth for EXISTENCE. Prunes recorded pods that no longer run,
    and adopts running `labeling-t-*` pods we didn't record (created from the
    console or another cwd) so they become usable — endpoint from the standard
    proxy URL, model key from the name suffix. Returns
    {"pods": state, "stale_removed": [ids], "adopted": [ids]}."""
    pods = load_pods()
    live_by_id = {p["id"]: p for p in live if p.get("id")}
    stale = [i for i in pods if i not in live_by_id]
    for i in stale:
        del pods[i]
    adopted = []
    for pid, p in live_by_id.items():
        name = str(p.get("name", ""))
        if pid in pods or not name.startswith(NAME_PREFIX + "-"):
            continue
        pods[pid] = {
            "id": pid,
            "model": name[len(NAME_PREFIX) + 1:].replace("-", "_"),
            "env_prefix": "",
            "endpoint": f"https://{pid}-8000.proxy.runpod.net",
            "gpu": None,
            "cost_per_hr": p.get("cost_per_hr", p.get("costPerHr")),
            "created_at": None,
            "terminate_after": None,  # unknown -> treated as unexpired
            "ready": True,            # it's running; a connect error corrects us
        }
        adopted.append(pid)
    if stale or adopted:
        _save(pods)
    return {"pods": pods, "stale_removed": stale, "adopted": adopted}


def resolve_endpoint(spec, explicit: str | None = None) -> str:
    """The endpoint an inference command should talk to for `spec` (see the
    module docstring for precedence). '' when nothing resolves — the caller
    owns the error message. Never queries RunPod: the inference path must not
    depend on runpodctl."""
    if explicit:
        return explicit.rstrip("/")
    candidates = [
        p for p in load_pods().values()
        if p.get("model") == spec.key and p.get("endpoint") and not _expired(p)
    ]
    if candidates:
        newest = max(candidates, key=lambda p: p.get("created_at") or "")
        return str(newest["endpoint"]).rstrip("/")
    env_endpoint = os.environ.get(f"{spec.env_prefix}_ENDPOINT", "").strip()
    if env_endpoint:
        print(
            f"note: using {spec.env_prefix}_ENDPOINT from the environment "
            f"(deprecated — `labeling-t-runpod up --model {spec.key}` records the "
            "endpoint in .labeling-t/pods.json; or pass --endpoint)",
            file=sys.stderr,
        )
        return env_endpoint.rstrip("/")
    return (spec.default_endpoint or "").rstrip("/")
