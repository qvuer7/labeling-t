"""Web orchestration layer: dataset overview, the upload stage end-to-end over
local storage, the in-process job runner, and LS-project-id persistence.

Stages that need a live vLLM/Label Studio/S3 are out of scope here (covered by
the pipeline-level tests); these exercise the FastAPI plumbing and job lifecycle.
"""

import io
import time

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from labeling_t.web import lsstate  # noqa: E402
from labeling_t.web.app import create_app  # noqa: E402
from labeling_t.web.jobs import JobRegistry  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate storage to a temp 'data/' dir (no S3): the layout defaults to
    # local 'data' when S3_BUCKET is unset, resolved relative to cwd.
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app())


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (12, 8), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _wait_job(client, job_id, want="done", tries=100):
    for _ in range(tries):
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error"):
            assert snap["status"] == want, snap
            return snap
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_models_endpoint_lists_specs(client):
    keys = [m["key"] for m in client.get("/api/models").json()]
    assert "qwen3_vl" in keys


def test_datasets_empty(client):
    assert client.get("/api/datasets").json() == []


def test_upload_writes_frames_and_shows_in_overview(client, tmp_path):
    files = [("files", ("a.png", _png_bytes(), "image/png")),
             ("files", ("b.png", _png_bytes(), "image/png"))]
    r = client.post("/api/upload", data={"dataset": "demo", "group": "all"}, files=files)
    assert r.status_code == 200
    snap = _wait_job(client, r.json()["id"])
    assert snap["result"]["uploaded"] == 2

    # frames landed under the local data layout
    frames = list((tmp_path / "data" / "datasets" / "demo" / "frames" / "all").glob("*.png"))
    assert len(frames) == 2

    # and the dataset overview reflects them
    overview = client.get("/api/datasets").json()
    assert len(overview) == 1
    groups = overview[0]["manifest"]["groups"]
    assert groups["all"]["frames"] == 2


def test_upload_rejects_non_images(client):
    files = [("files", ("notes.txt", b"hello", "text/plain"))]
    r = client.post("/api/upload", data={"dataset": "demo", "group": "all"}, files=files)
    assert r.status_code == 400


def test_job_not_found(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_job_registry_runs_and_reports_progress():
    reg = JobRegistry()

    def fn(job):
        job.set_progress(2, 2)
        job.log("did work")
        return {"ok": True}

    job = reg.submit("test", fn)
    for _ in range(100):
        if reg.get(job.id).snapshot()["status"] == "done":
            break
        time.sleep(0.02)
    snap = reg.get(job.id).snapshot()
    assert snap["status"] == "done"
    assert snap["result"] == {"ok": True}
    assert snap["done"] == 2 and snap["total"] == 2
    assert "did work" in snap["log"]


def test_job_registry_captures_errors():
    reg = JobRegistry()

    def boom(job):
        raise ValueError("nope")

    job = reg.submit("test", boom)
    for _ in range(100):
        if reg.get(job.id).snapshot()["status"] == "error":
            break
        time.sleep(0.02)
    snap = reg.get(job.id).snapshot()
    assert snap["status"] == "error"
    assert "nope" in snap["error"]


def test_lsstate_roundtrip(tmp_path, monkeypatch):
    monkeypatch.delenv("S3_BUCKET", raising=False)
    monkeypatch.chdir(tmp_path)
    assert lsstate.get_project_id("demo", "all") is None
    lsstate.set_project_id("demo", "all", 42)
    lsstate.set_project_id("demo", "g2", 7)
    assert lsstate.get_project_id("demo", "all") == 42
    assert lsstate.load("demo") == {"all": 42, "g2": 7}
