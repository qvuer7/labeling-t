"""runpod CLI: --json envelopes + pod state integration (runpodctl stubbed).

The state contract under test: `up` records the pod BEFORE waiting on readiness
(a timeout must not lose a billing pod), `down` removes it, `status` reconciles
(prunes dead ids, adopts foreign labeling-t-* pods) — and nothing ever writes
.env (that file is secrets only).
"""

from __future__ import annotations

import json

import httpx

import labeling_t.runpod as rp
from labeling_t import podstate


def _fake_runpodctl(responses: dict):
    """A _runpodctl stub keyed on the subcommand pair, e.g. ('pod', 'create')."""
    def fake(args, env):
        key = (args[0], args[1] if len(args) > 1 and not args[1].startswith("-") else "")
        if key in responses:
            return responses[key]
        if (args[0], "") in responses:
            return responses[(args[0], "")]
        raise AssertionError(f"unexpected runpodctl call: {args}")
    return fake


# ---- --json envelope (moved from test_server_adapters.py) -------------------

def test_status_json_envelope(monkeypatch, capsys):
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
    monkeypatch.setattr(rp, "_runpodctl", lambda args, env: json.dumps(
        {"clientBalance": 1.0, "currentSpendPerHr": 0} if args[0] == "user" else []))
    rc = rp.main(["status"])
    out, err = capsys.readouterr()
    assert rc == 0 and "no pods running" in out and not err


def test_gpus_json_envelope(capsys):
    rc = rp.main(["gpus", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["ok"] is True
    keys = {g["key"] for g in envelope["result"]["gpus"]}
    assert "a40" in keys and "rtx4090" in keys
    assert envelope["result"]["default"] == "rtx4090"


def test_datacenters_json_error_when_no_stock(monkeypatch, capsys):
    monkeypatch.setattr(rp, "_datacenters_with_stock", lambda gpu_id, env: [])
    rc = rp.main(["datacenters", "--gpu", "a40", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 1 and envelope["ok"] is False
    assert envelope["error"]["gpu_id"] == "NVIDIA A40"  # ok mirrors the exit code


def test_runpodctl_failure_becomes_error_envelope(monkeypatch, capsys):
    def boom(args, env):
        raise SystemExit("runpodctl user failed:\nboom")

    monkeypatch.setattr(rp, "_runpodctl", boom)
    rc = rp.main(["status", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 1 and envelope["ok"] is False and "boom" in envelope["error"]["message"]


# ---- pod state integration ---------------------------------------------------

def test_up_records_state_and_never_touches_env_file(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("SECRET_KEY=shhh\n")
    monkeypatch.setattr(rp, "_runpodctl", _fake_runpodctl({
        ("pod", "list"): "[]",
        ("datacenter", "list"): "[]",
        ("pod", "create"): json.dumps({"id": "podX", "costPerHr": 0.39}),
    }))
    rc = rp.main(["up", "--model", "sam2", "--gpu", "a40", "--no-wait", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["ok"] is True
    assert envelope["result"]["id"] == "podX"
    assert envelope["result"]["endpoint"] == "https://podX-8000.proxy.runpod.net"
    # state recorded even though we never waited for readiness
    pods = podstate.load_pods()
    assert pods["podX"]["model"] == "sam2" and pods["podX"]["ready"] is False
    assert pods["podX"]["endpoint"] == envelope["result"]["endpoint"]
    assert pods["podX"]["terminate_after"] == envelope["result"]["terminate_after"]
    # .env is a SECRETS file — provisioning must not rewrite it
    assert (tmp_path / ".env").read_text() == "SECRET_KEY=shhh\n"


def test_up_marks_ready_in_state_after_health_ok(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rp, "_runpodctl", _fake_runpodctl({
        ("pod", "list"): "[]",
        ("datacenter", "list"): "[]",
        ("pod", "create"): json.dumps({"id": "podY", "costPerHr": 0.39}),
    }))
    monkeypatch.setattr(rp.httpx, "get", lambda url, timeout: httpx.Response(
        200, json={"ready": True, "model": "sam2"},
        request=httpx.Request("GET", url)))
    rc = rp.main(["up", "--model", "sam2", "--gpu", "a40", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["result"]["ready"] is True
    assert podstate.load_pods()["podY"]["ready"] is True


def test_down_removes_state(monkeypatch, capsys):
    podstate.record_pod({"id": "p1", "model": "sam2", "env_prefix": "SAM2",
                         "endpoint": "https://p1-8000.proxy.runpod.net", "gpu": None,
                         "cost_per_hr": 0.4, "created_at": None,
                         "terminate_after": None, "ready": True})
    monkeypatch.setattr(rp, "_runpodctl", _fake_runpodctl({
        ("pod", "list"): json.dumps([{"id": "p1", "name": "labeling-t-sam2",
                                      "costPerHr": 0.4, "desiredStatus": "RUNNING"}]),
        ("pod", "delete"): "",
    }))
    rc = rp.main(["down", "p1", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["result"]["deleted"] == [{"id": "p1", "name": "labeling-t-sam2"}]
    assert podstate.load_pods() == {}


def test_status_prunes_dead_and_enriches_live(monkeypatch, capsys):
    podstate.record_pod({"id": "dead", "model": "sam2", "env_prefix": "SAM2",
                         "endpoint": "https://dead-8000.proxy.runpod.net", "gpu": None,
                         "cost_per_hr": 0.4, "created_at": None,
                         "terminate_after": None, "ready": True})
    monkeypatch.setattr(rp, "_runpodctl", _fake_runpodctl({
        ("user", ""): json.dumps({"clientBalance": 5.0, "currentSpendPerHr": 0.7}),
        ("pod", "list"): json.dumps([{"id": "l1", "name": "labeling-t-owlv2",
                                      "costPerHr": 0.7, "desiredStatus": "RUNNING"}]),
    }))
    rc = rp.main(["status", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["result"]["stale_removed"] == ["dead"]
    (pod,) = envelope["result"]["pods"]
    # foreign labeling-t-* pod adopted: model from the name, endpoint from the proxy
    assert pod["model"] == "owlv2"
    assert pod["endpoint"] == "https://l1-8000.proxy.runpod.net"
    state = podstate.load_pods()
    assert set(state) == {"l1"}  # dead pruned, live adopted


# ---- guardrails: duplicate refusal + --budget ---------------------------------

def test_up_refuses_duplicate_pod_with_recovery_payload(monkeypatch, capsys):
    monkeypatch.setattr(rp, "_runpodctl", _fake_runpodctl({
        ("pod", "list"): json.dumps([{"id": "old1", "name": "labeling-t-sam2",
                                      "costPerHr": 0.4, "desiredStatus": "RUNNING"}]),
    }))
    rc = rp.main(["up", "--model", "sam2", "--gpu", "a40", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 1 and envelope["ok"] is False
    # the agent's correct next move rides in the payload: reuse this endpoint
    assert envelope["error"]["existing"] == {
        "id": "old1", "endpoint": "https://old1-8000.proxy.runpod.net", "cost_per_hr": 0.4}
    assert "--force" in envelope["error"]["message"]


def test_up_force_overrides_duplicate_refusal(monkeypatch, capsys):
    monkeypatch.setattr(rp, "_runpodctl", _fake_runpodctl({
        ("pod", "list"): json.dumps([{"id": "old1", "name": "labeling-t-sam2",
                                      "costPerHr": 0.4, "desiredStatus": "RUNNING"}]),
        ("datacenter", "list"): "[]",
        ("pod", "create"): json.dumps({"id": "new1", "costPerHr": 0.4}),
    }))
    rc = rp.main(["up", "--model", "sam2", "--gpu", "a40", "--force", "--no-wait", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 0 and envelope["result"]["id"] == "new1"


def test_up_budget_caps_hours_when_price_is_known(monkeypatch, capsys):
    monkeypatch.setattr(rp, "_runpodctl", _fake_runpodctl({
        ("pod", "list"): "[]",
        # hypothetical future runpodctl that DOES report prices
        ("gpu", "list"): json.dumps([{"gpuId": "NVIDIA A40", "securePrice": 0.5}]),
        ("datacenter", "list"): "[]",
        ("pod", "create"): json.dumps({"id": "p1", "costPerHr": 0.5}),
    }))
    rc = rp.main(["up", "--model", "sam2", "--gpu", "a40",
                  "--hours", "10", "--budget", "1.0", "--no-wait", "--json"])
    out, err = capsys.readouterr()
    envelope = json.loads(out)
    assert rc == 0 and "caps runtime at 2.0h" in err
    # terminate_after reflects the capped 2h, not the requested 10h
    from datetime import datetime, timezone
    term = datetime.strptime(envelope["result"]["terminate_after"],
                             "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    left = (term - datetime.now(timezone.utc)).total_seconds() / 3600
    assert 1.9 < left <= 2.01


def test_up_over_budget_deletes_pod_and_suggests_hours(monkeypatch, capsys):
    deleted = []

    def fake(args, env):
        if args[:2] == ["pod", "delete"]:
            deleted.append(args[2])
            return ""
        return _fake_runpodctl({
            ("pod", "list"): "[]",
            ("gpu", "list"): "[]",   # real runpodctl: no prices -> no pre-cap
            ("datacenter", "list"): "[]",
            ("pod", "create"): json.dumps({"id": "exp1", "costPerHr": 2.0}),
        })(args, env)

    monkeypatch.setattr(rp, "_runpodctl", fake)
    rc = rp.main(["up", "--model", "sam2", "--gpu", "a40",
                  "--hours", "3", "--budget", "1.0", "--json"])
    envelope = json.loads(capsys.readouterr().out)
    assert rc == 1 and envelope["ok"] is False
    assert deleted == ["exp1"]                      # billing stopped immediately
    assert podstate.load_pods() == {}               # never entered usable state
    assert envelope["error"]["cost_per_hr"] == 2.0
    assert envelope["error"]["suggested_hours"] == 0.5
    assert "--hours 0.5" in envelope["error"]["message"]
