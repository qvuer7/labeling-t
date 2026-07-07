"""Shared test hermeticity.

Pod runtime state lives in a cwd-relative file (.labeling-t/pods.json) resolved
through $LABELING_T_PODS (podstate.state_path). Point every test at a tmp file
so no test reads a real dev state file — and none can write one.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_pod_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LABELING_T_PODS", str(tmp_path / "pods.json"))
