# labeling-t

Batch auto-labeling backend: run a vision model over images once, get pre-labels,
verify the uncertain ones in Label Studio, export to a training format. The model
carries most of the labeling load; a human only corrects.

**Design rule:** an owned **neutral schema** is the contract. The model
(Qwen3-VL), Label Studio, and COCO are all swappable adapters hanging off it —
the tool is never trapped by any single dependency.

Full architecture, roadmap, and the local-vs-cloud gap: see **[doc.md](doc.md)**.

## Pipeline

```
raw frames → vLLM (rented GPU) → neutral-schema JSON → Label Studio (verify) → COCO
```

The model runs on a **remote GPU**; this machine has no GPU/torch — it only POSTs
to a vLLM OpenAI-compatible endpoint over HTTP.

## Status

v0, working end-to-end on real basketball footage. Deliberately narrow scope:
**boxes only, images only, one model** (masks / tracking / video are v1).

- **Model:** Qwen3-VL-8B on vLLM, served on a rented GPU (RunPod).
- **Verification:** Label Studio, auto-generated config + pre-annotations, round-trip back to the schema.
- **Cloud:** S3 / DigitalOcean Spaces storage foundation in place (presigned URLs); full cloud loop in progress (see doc.md §8 + roadmap).
- **72 tests passing.**

## Install

```bash
uv sync --extra integrations --extra cloud
uv run pytest -q
```

## Commands

```bash
# 1. serve a model on a rented GPU (RunPod)
labeling-t-runpod gpus
labeling-t-runpod up --model qwen3_vl --gpu rtx4090   # rents, serves, writes endpoint -> .env
labeling-t-runpod down                                 # stop billing

# 2. label a folder of frames  (image + prompt -> vLLM -> neutral-schema JSON)
uv run python scripts/spike.py --model qwen3_vl --images data/spike_frames --out labels
#   --check : endpoint up?     --raw : dump raw model output (tuning a new model)

# 3. human verification (Label Studio)
docker compose up -d                       # LS :8080 + image server :8081
uv run python scripts/ls_setup.py          # -> API token
labeling-t import-ls --labels labels --api-key <token> \
    --project demo --categories player,ball,referee \
    --image-base-url http://localhost:8081 --image-root data

# 4. after correcting boxes in the UI
labeling-t from-ls --export export.json --out verified/
labeling-t to-coco  --labels verified/ --out dataset.coco.json
```

Label Studio UI: http://localhost:8080 (`admin@labeling-t.local` / `labeling-t-admin`).

## Web UI

A browser console drives the same loop — upload → auto-label → send to Label
Studio → pull verified — with live progress, plus a RunPod GPU panel. It's a thin
FastAPI adapter over the pipeline functions; nothing new about the schema.

```bash
uv sync --extra web --extra integrations --extra cloud
docker compose up -d            # Label Studio :8080 (separate, as above)
labeling-t-web                  # -> http://127.0.0.1:8000
```

It reads `LS_URL` / `LS_API_KEY` from `.env` so the token isn't retyped per
action (the local compose ships a fixed dev token). Local operator tool: binds
`127.0.0.1`, no auth — front it with Caddy + basic-auth if you expose it.

## Layout

```
src/labeling_t/      the framework (src-layout, pip-installable)
  schema · geometry · models · gpu · runpod · model_client · prelabel · storage
  adapters/{label_studio,coco} · ingest · verify · cli · config
  web/                 FastAPI app + static SPA (labeling-t-web)
scripts/             spike.py · ls_setup.py · serve_vllm.sh
tests/               103 tests
docker-compose.yml + nginx/   local Label Studio + CORS image server
```

Console commands: `labeling-t` (prelabel / import-ls / from-ls / to-coco / ls-config),
`labeling-t-runpod` (up / down / status / gpus), and `labeling-t-web` (browser console).
