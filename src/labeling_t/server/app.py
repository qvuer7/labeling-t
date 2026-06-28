"""FastAPI model-server: ONE uniform contract, per-model dispatch behind it.

    POST /infer  {image_url, queries[], params{}}  ->  {width, height, detections[]}
    GET  /health                                   ->  {status, model, ready}

`MODEL` (env) selects the adapter the pod loads (default "stub" = no torch, for
the seam test + CI). The wire contract lives in contract.py; per-model logic
lives in adapters/. Adding a model changes neither this file nor the client.

Data flow:
    /infer ─► adapter.detect(image_url, queries, params) ─► InferResponse
              (real adapter: fetch image_url, run model, normalize to abs-px xyxy)
"""

from __future__ import annotations

import os

try:
    from fastapi import FastAPI, HTTPException
except ImportError as exc:  # pragma: no cover - serve needs the [models] (or [web]) extra
    raise SystemExit("the model-server needs FastAPI: `uv sync --extra models`") from exc

from .adapters import get_adapter
from .contract import InferRequest, InferResponse


def create_app() -> FastAPI:
    model = os.environ.get("MODEL", "stub")
    adapter = get_adapter(model, os.environ.get("HF_MODEL"))
    adapter.load()  # blocks startup until weights are ready -> /health reachable == ready

    app = FastAPI(title="labeling-t model-server", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model, "ready": adapter.ready}

    @app.post("/infer", response_model=InferResponse)
    def infer(req: InferRequest) -> InferResponse:
        if not adapter.ready:
            raise HTTPException(status_code=503, detail="model still loading")
        return adapter.detect(req.image_url, req.queries, req.params)

    return app


def main() -> int:  # pragma: no cover - process entrypoint
    import uvicorn

    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
