"""FastAPI app + `labeling-t-web` entry point.

Exposes the pipeline stages the operator drives by hand today — upload frames,
auto-label, push to Label Studio, pull verified labels back — plus a RunPod GPU
panel, as a small JSON API behind a static SPA. Every stage reuses the existing
core functions; nothing here re-implements pipeline logic.

Long stages run as background Jobs (see jobs.py); the browser polls
GET /api/jobs/{id} for live progress. Label Studio creds default from the
environment (LS_URL / LS_API_KEY) so they aren't retyped per action.

Local operator tool: binds 127.0.0.1, no auth. If you expose it, front it with
the existing Caddy + basic-auth.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import load_env
from . import datasets as datasets_mod
from . import lsstate
from .jobs import Job, JobRegistry

_STATIC = Path(__file__).parent / "static"
_IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def ls_creds() -> tuple[str, str]:
    """Label Studio base URL + API token from the environment."""
    return os.environ.get("LS_URL", "http://localhost:8080"), os.environ.get("LS_API_KEY", "")


# --- request bodies -----------------------------------------------------------

class PrelabelReq(BaseModel):
    dataset: str
    group: str
    model: str = "qwen3_vl"
    categories: list[str] | None = None
    min_score: float = 0.0
    concurrency: int = 8


class ImportLSReq(BaseModel):
    dataset: str
    group: str
    categories: list[str]
    ttl: int = 604800  # 7 days of presigned-URL life


class VerifyReq(BaseModel):
    dataset: str
    group: str


class GpuUpReq(BaseModel):
    model: str = "qwen3_vl"
    gpu: str = "rtx4090"
    hours: float = 3.0


class GpuDownReq(BaseModel):
    pods: list[str] | None = None
    all: bool = False


def create_app() -> FastAPI:
    load_env()  # LS_URL / LS_API_KEY / S3_* / model endpoints from .env
    app = FastAPI(title="labeling-t web")
    jobs = JobRegistry()
    app.state.jobs = jobs

    # --- datasets / models (read-only) ---------------------------------------

    @app.get("/api/models")
    def list_models() -> list[dict]:
        from ..models import REGISTRY

        return [
            {"key": s.key, "name": s.name, "categories": list(s.categories)}
            for s in REGISTRY.values()
        ]

    @app.get("/api/datasets")
    def list_datasets() -> list[dict]:
        return datasets_mod.list_datasets()

    @app.get("/api/datasets/{dataset}")
    def get_dataset(dataset: str) -> dict:
        return datasets_mod.dataset_overview(dataset)

    # --- stage 1: upload images ----------------------------------------------

    @app.post("/api/upload")
    async def upload(
        dataset: str = Form(...),
        group: str = Form(...),
        files: list[UploadFile] = File(...),
    ) -> dict:
        from ..ingest import ingest_images
        from ..layout import DatasetLayout
        from ..manifest import build_manifest
        from ..storage import open_storage

        # Read uploads to a temp dir now (UploadFile is tied to this request);
        # the background job then ingests from disk into the dataset's frames.
        tmp = Path(tempfile.mkdtemp(prefix="labeling-t-upload-"))
        saved = 0
        for f in files:
            name = Path(f.filename or "").name
            if not name or Path(name).suffix.lower() not in _IMG_SUFFIXES:
                continue
            (tmp / name).write_bytes(await f.read())
            saved += 1
        if saved == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(400, "no image files in upload")

        dest = DatasetLayout.from_env(dataset).frames(group)

        def run(job: Job) -> dict:
            try:
                uploaded, total = ingest_images(
                    str(tmp), dest, storage=open_storage(dest),
                    on_progress=job.set_progress,
                )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            build_manifest(dataset, base=None)
            return {"uploaded": uploaded, "total": total, "dest": dest}

        return jobs.submit("upload", run).snapshot()

    # --- stage 2: auto-label --------------------------------------------------

    @app.post("/api/jobs/prelabel")
    def prelabel(req: PrelabelReq) -> dict:
        from ..layout import DatasetLayout
        from ..manifest import build_manifest
        from ..model_client import client_for
        from ..models import get_spec
        from ..prelabel import prelabel_cloud
        from ..storage import open_storage

        def run(job: Job) -> dict:
            spec = get_spec(req.model)
            layout = DatasetLayout.from_env(req.dataset)
            frames_prefix, labels_prefix = layout.frames(req.group), layout.labels(req.group)
            storage = open_storage(frames_prefix)
            frames = [u for u in storage.list(frames_prefix + "/")
                      if u.lower().endswith(_IMG_SUFFIXES)]
            if not frames:
                raise ValueError(f"no frames under {frames_prefix} (upload first)")
            job.log(f"labeling {len(frames)} frames with {spec.name}")
            client = client_for(spec, categories=req.categories or None)
            with client:
                n = prelabel_cloud(
                    frames, client, labels_prefix, storage=storage,
                    min_score=req.min_score, max_concurrency=req.concurrency,
                    on_progress=job.set_progress,
                )
            job.failures = len(frames) - n
            build_manifest(req.dataset, base=None)
            return {"labeled": n, "total": len(frames), "labels_prefix": labels_prefix}

        return jobs.submit("prelabel", run).snapshot()

    # --- stage 3: send to Label Studio ---------------------------------------

    @app.post("/api/jobs/import-ls")
    def import_ls(req: ImportLSReq) -> dict:
        from ..adapters.label_studio import import_to_label_studio
        from ..layout import DatasetLayout
        from ..schema import ImageLabels
        from ..storage import open_storage

        url, key = ls_creds()
        if not key:
            raise HTTPException(400, "LS_API_KEY is not set (.env)")

        def run(job: Job) -> dict:
            labels_prefix = DatasetLayout.from_env(req.dataset).labels(req.group)
            storage = open_storage(labels_prefix)
            uris = [u for u in storage.list(labels_prefix + "/") if u.endswith(".json")]
            if not uris:
                raise ValueError(f"no labels under {labels_prefix} (auto-label first)")
            images = [ImageLabels.model_validate_json(storage.read_bytes(u).decode()) for u in uris]
            job.set_progress(len(images), len(images))
            job.log(f"importing {len(images)} tasks into Label Studio")
            project = import_to_label_studio(
                images, base_url=url, api_key=key,
                project_title=f"{req.dataset}-{req.group}", categories=req.categories,
                presign=lambda uri: storage.presigned_url(uri, req.ttl),
            )
            lsstate.set_project_id(req.dataset, req.group, project.id, base=None)
            open_url = f"{url.rstrip('/')}/projects/{project.id}/data"
            job.log(f"open in Label Studio: {open_url}")
            return {"project_id": project.id, "tasks": len(images), "open_url": open_url}

        return jobs.submit("import-ls", run).snapshot()

    # --- stage 4: pull verified labels ---------------------------------------

    @app.post("/api/jobs/verify")
    def verify(req: VerifyReq) -> dict:
        from ..manifest import build_manifest
        from ..verify import pull_verified

        url, key = ls_creds()
        if not key:
            raise HTTPException(400, "LS_API_KEY is not set (.env)")

        def run(job: Job) -> dict:
            project_id = lsstate.get_project_id(req.dataset, req.group, base=None)
            if project_id is None:
                raise ValueError("no Label Studio project for this dataset/group "
                                 "(run 'Send to Label Studio' first)")
            job.log(f"pulling verified labels from LS project {project_id}")
            n = pull_verified(
                req.dataset, req.group, url=url, api_key=key,
                project_id=project_id, base=None, on_progress=job.set_progress,
            )
            build_manifest(req.dataset, base=None)
            return {"verified": n, "project_id": project_id}

        return jobs.submit("verify", run).snapshot()

    # --- job polling ----------------------------------------------------------

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.snapshot()

    # --- GPU (RunPod) ---------------------------------------------------------

    @app.get("/api/gpu/status")
    def gpu_status() -> dict:
        from ..runpod import list_pods_with_balance

        return list_pods_with_balance()

    @app.post("/api/gpu/up")
    def gpu_up(req: GpuUpReq) -> dict:
        from ..runpod import start_pod

        def run(job: Job) -> dict:
            return start_pod(req.model, gpu=req.gpu, hours=req.hours, wait=True, log=job.log)

        return jobs.submit("gpu-up", run).snapshot()

    @app.post("/api/gpu/down")
    def gpu_down(req: GpuDownReq) -> dict:
        from ..runpod import AmbiguousPods, stop_pods

        try:
            deleted = stop_pods(pods=req.pods or None, all=req.all)
        except AmbiguousPods as exc:
            raise HTTPException(409, {"message": str(exc), "pods": exc.pods})
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"deleted": deleted}

    # --- static SPA -----------------------------------------------------------

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    app.mount("/", StaticFiles(directory=str(_STATIC)), name="static")
    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("LABELING_T_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("LABELING_T_WEB_PORT", "8000"))
    print(f"labeling-t web -> http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
