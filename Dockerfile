# labeling-t model-server image (the `[models]` extra). Runs ON the RunPod GPU.
#
# torch pulls its own CUDA (cu13) via nvidia-* wheels, so a SLIM base is enough —
# no nvcc / CUDA toolkit needed. This holds for SAM2 too: we use transformers'
# NATIVE Sam2 (plain torch), NOT facebookresearch/sam2 (which compiles a custom
# CUDA `_C` ext and would force a cuda-devel base). All adapters share this image.
#
# Multi-stage by LAYER: a server-code change rebuilds only the thin app layer,
# not the multi-GB torch+CUDA deps layer (keeps the manual build/push loop sane).
#
# Build + push (public GHCR -> RunPod needs no registry creds):
#   docker build -t ghcr.io/qvuer7/labeling-t-models:0.1 .
#   docker push  ghcr.io/qvuer7/labeling-t-models:0.1
#
# Pod runs `labeling-t-models`, reading MODEL / HF_MODEL / HF_TOKEN / PORT from env.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/weights

# libgomp1: OpenMP runtime torch needs. (No CUDA toolkit — torch wheels carry it.)
# libglib2.0-0: cv2 (opencv-python-headless) needs libgthread; LocateAnything's
# processor imports cv2 at module top, so without it from_pretrained ImportErrors.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app

# --- deps layer (cached; invalidated only when pyproject/uv.lock change) ---
COPY pyproject.toml uv.lock ./
RUN uv sync --extra models --no-dev --no-install-project

# --- app layer (thin; rebuilt on every code change) ---
COPY README.md ./
COPY src/ ./src/
RUN uv sync --extra models --no-dev

ENV MODEL=owlv2 \
    PORT=8000
EXPOSE 8000

# exec the installed console script directly (NOT `uv run`, which re-syncs the
# venv on every container start -> slow, fragile cold starts on the pod).
CMD ["/app/.venv/bin/labeling-t-models"]
