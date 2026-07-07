"""Pod runtime state: persistence, reconciliation, endpoint resolution.

The state file is the boundary between provisioning (runpod.py writes it) and
inference (model_client resolves from it) — these tests pin the contract at
that boundary: precedence order, expiry, newest-wins, adopt/prune, and the
never-crash rule for a corrupt file. conftest.py points LABELING_T_PODS at a
tmp file for every test.
"""

from __future__ import annotations

import json
from pathlib import Path

from labeling_t import podstate
from labeling_t.models import ModelSpec

SPEC = ModelSpec(key="sam2", name="sam2", env_prefix="SAM2", prompt="",
                 backend="transformers")
SAAS = ModelSpec(key="openai_ocr", name="gpt-4o-mini", env_prefix="OPENAI", prompt="",
                 backend="openai", default_endpoint="https://api.openai.com/v1")

FUTURE = "2099-01-01T00:00:00Z"
PAST = "2000-01-01T00:00:00Z"


def _entry(pod_id: str, model: str = "sam2", *, endpoint: str | None = None,
           created_at: str = "2026-07-07T00:00:00Z", terminate_after: str = FUTURE) -> dict:
    return {"id": pod_id, "model": model, "env_prefix": "SAM2",
            "endpoint": endpoint or f"https://{pod_id}-8000.proxy.runpod.net",
            "gpu": "NVIDIA A40", "cost_per_hr": 0.4, "created_at": created_at,
            "terminate_after": terminate_after, "ready": True}


# ---- persistence -----------------------------------------------------------

def test_record_load_remove_roundtrip():
    podstate.record_pod(_entry("p1"))
    podstate.record_pod(_entry("p2"))
    assert set(podstate.load_pods()) == {"p1", "p2"}
    # upsert, not append: re-recording an id replaces its entry
    podstate.record_pod({**_entry("p1"), "ready": False})
    assert podstate.load_pods()["p1"]["ready"] is False
    assert podstate.remove_pods(["p1", "nope"]) == ["p1"]
    assert set(podstate.load_pods()) == {"p2"}


def test_missing_and_corrupt_file_load_as_empty(capsys):
    assert podstate.load_pods() == {}  # no file yet
    path = podstate.state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert podstate.load_pods() == {}  # cache, not truth: never blocks a command
    assert "unreadable pod state" in capsys.readouterr().err
    podstate.record_pod(_entry("p1"))  # and it heals on the next write
    assert set(podstate.load_pods()) == {"p1"}


def test_state_file_is_versioned_json(monkeypatch):
    podstate.record_pod(_entry("p1"))
    data = json.loads(Path(podstate.state_path()).read_text())
    assert data["version"] == 1 and "p1" in data["pods"]


# ---- resolve_endpoint precedence (one test per boundary) --------------------

def test_explicit_endpoint_beats_recorded_pod(monkeypatch):
    podstate.record_pod(_entry("p1"))
    assert podstate.resolve_endpoint(SPEC, "http://gpu:9000/") == "http://gpu:9000"


def test_recorded_pod_beats_env(monkeypatch):
    monkeypatch.setenv("SAM2_ENDPOINT", "http://from-env:8000")
    podstate.record_pod(_entry("p1"))
    assert podstate.resolve_endpoint(SPEC) == "https://p1-8000.proxy.runpod.net"


def test_env_beats_spec_default_and_warns_deprecated(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_ENDPOINT", "http://proxy:9000")
    assert podstate.resolve_endpoint(SAAS) == "http://proxy:9000"
    assert "deprecated" in capsys.readouterr().err


def test_spec_default_when_nothing_else(monkeypatch):
    monkeypatch.delenv("OPENAI_ENDPOINT", raising=False)
    assert podstate.resolve_endpoint(SAAS) == "https://api.openai.com/v1"


def test_empty_when_nothing_resolves(monkeypatch):
    monkeypatch.delenv("SAM2_ENDPOINT", raising=False)
    assert podstate.resolve_endpoint(SPEC) == ""


# ---- resolve_endpoint selection among recorded pods --------------------------

def test_newest_unexpired_pod_wins(monkeypatch):
    podstate.record_pod(_entry("old", created_at="2026-07-01T00:00:00Z"))
    podstate.record_pod(_entry("new", created_at="2026-07-07T00:00:00Z"))
    assert podstate.resolve_endpoint(SPEC) == "https://new-8000.proxy.runpod.net"


def test_expired_pod_is_skipped(monkeypatch):
    monkeypatch.setenv("SAM2_ENDPOINT", "http://from-env:8000")
    podstate.record_pod(_entry("dead", terminate_after=PAST))
    assert podstate.resolve_endpoint(SPEC) == "http://from-env:8000"


def test_other_models_pods_dont_match(monkeypatch):
    monkeypatch.delenv("SAM2_ENDPOINT", raising=False)
    podstate.record_pod(_entry("p1", model="owlv2"))
    assert podstate.resolve_endpoint(SPEC) == ""


def test_adopted_pod_without_terminate_after_counts_as_unexpired():
    podstate.record_pod({**_entry("p1"), "terminate_after": None, "created_at": None})
    assert podstate.resolve_endpoint(SPEC) == "https://p1-8000.proxy.runpod.net"


# ---- reconcile: live list is the truth for existence -------------------------

def test_reconcile_prunes_dead_and_adopts_foreign():
    podstate.record_pod(_entry("dead"))
    podstate.record_pod(_entry("alive"))
    live = [
        {"id": "alive", "name": "labeling-t-sam2", "cost_per_hr": 0.4},
        {"id": "foreign", "name": "labeling-t-locate-anything", "cost_per_hr": 0.7},
        {"id": "unrelated", "name": "someone-elses-pod", "cost_per_hr": 1.0},
    ]
    rec = podstate.reconcile(live)
    assert rec["stale_removed"] == ["dead"] and rec["adopted"] == ["foreign"]
    pods = podstate.load_pods()
    assert set(pods) == {"alive", "foreign"}  # unrelated pods are NOT adopted
    # adoption recovers the model key (dashes -> underscores) + the proxy endpoint
    assert pods["foreign"]["model"] == "locate_anything"
    assert pods["foreign"]["endpoint"] == "https://foreign-8000.proxy.runpod.net"
    # our own recorded entry survives untouched (not overwritten by adoption)
    assert pods["alive"]["created_at"] == "2026-07-07T00:00:00Z"


def test_reconcile_empty_live_clears_state():
    podstate.record_pod(_entry("p1"))
    rec = podstate.reconcile([])
    assert rec["stale_removed"] == ["p1"] and podstate.load_pods() == {}
