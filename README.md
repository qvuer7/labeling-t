# labeling-t

Batch auto-labeling backend: run a vision model over images once, get pre-labels,
verify the uncertain ones in Label Studio, export to a training format. The model
carries most of the labeling load; a human only corrects.

**Design rule:** an owned **neutral schema** is the contract. The models, Label
Studio, and COCO are all swappable adapters hanging off it — the tool is never
trapped by any single dependency.

Full architecture: **[doc.md](doc.md)** · roadmap: **[plans/roadmap.md](plans/roadmap.md)**.

## Pipeline

```
raw frames → model (rented GPU or hosted API) → neutral-schema JSON → Label Studio → COCO
                              ├─ optional: detector boxes → SAM2 → masks   (segment-cloud)
                              └─ optional: region crops → hosted VLM → text (transcribe, OCR)
```

Inference runs **remotely** (a rented GPU or a hosted API); this machine has no
GPU/torch — it only POSTs over HTTP.

## Status

v0.x, working end-to-end on real basketball footage. **Boxes + masks, images only.**

- **Models (one registry, 3 backends):**
  - `transformers` (our model-server, GPU pod): **OWLv2**, **LocateAnything-3B**, **SAM2** (segmenter)
  - `vllm` (stock image, GPU pod): **Qwen3-VL-8B**
  - hosted API (no GPU): **GPT-4o**, **Gemini**
- **Two-stage:** any detector's boxes → SAM2 → per-instance masks (`segment-cloud`, in-place enrichment).
- **OCR:** crop any labeled region → hosted VLM reads it → `Detection.text` (`transcribe[-cloud]`),
  with per-detection resume so an interrupted run never re-bills.
- **Verification:** Label Studio, auto-generated config + pre-annotations (boxes, polygon or brush
  masks), round-trip back to the schema.
- **Cloud:** S3 / DigitalOcean Spaces, presigned URLs, full cloud loop (`prelabel-cloud` → `import-ls-cloud` → `from-ls-cloud`).
- **157 tests passing.**

## Install

```bash
uv sync --extra integrations --extra cloud
uv run pytest -q
```

## Commands

Models live in a registry (`src/labeling_t/models.py`); pick one with `--model`:
`owlv2`, `locate_anything`, `sam2`, `qwen3_vl`, `openai_vl`, `gemini_vl`,
`openai_ocr`, `gemini_ocr`.

### Full lifecycle (images → boxes → masks → Label Studio)

The complete detect→segment→verify loop over a fresh image folder:

```bash
# 0. upload images into a dataset group
labeling-t ingest-images --src ./images --dataset my-dataset --group all

# 1. serve a detector on a rented GPU (RunPod). `up` auto-targets datacenters that
#    actually have the GPU in stock (see `datacenters`); hosted models need no pod.
labeling-t-runpod gpus                                       # GPU presets
labeling-t-runpod datacenters --gpu rtx5090                  # where a GPU is in stock
labeling-t-runpod up --model locate_anything --gpu a40 --budget 3   # rent + serve; endpoint -> .labeling-t/pods.json
#    guardrails: refuses a duplicate pod for the same model (--force overrides);
#    --budget $ hard-caps cost (over-budget pods are deleted with suggested --hours)

# 2. detect -> neutral-schema label JSONs land straight in S3 (use --concurrency 1
#    for the transformers backend). --labels-name keeps a 2nd model's labels separate.
labeling-t prelabel-cloud --dataset my-dataset --group all \
    --model locate_anything --categories cat,dog --concurrency 1

# 3. swap pods: detector down, segmenter up
labeling-t-runpod down --all
labeling-t-runpod up --model sam2 --gpu a40

# 4. boxes -> SAM2 masks; the SAME label files gain Detection.mask in place
#    (--to-name writes enriched copies instead; resume skips already-masked boxes)
labeling-t segment-cloud --dataset my-dataset --group all --model sam2
labeling-t-runpod down --all                                 # stop billing

# 5. import to a hosted Label Studio (frames via presigned S3 URLs; brush or
#    polygon mask pre-annotations)
labeling-t import-ls-cloud --dataset my-dataset --group all \
    --project "my-dataset masks" --categories cat,dog --masks --mask-format brush

# 6. after verifying in the UI, pull the truth back + export
labeling-t from-ls-cloud --dataset my-dataset --group all --project-id <ID>
labeling-t to-coco --labels verified/ --out dataset.coco.json
```

### OCR (region transcription, no GPU)

Read text out of labeled regions with a hosted VLM — the crop travels, the
result lands on the region's `Detection.text`, referenced to the whole frame:

```bash
labeling-t transcribe-cloud --dataset my-dataset --group all \
    --categories scoreboard --model openai_ocr \
    --prompt 'Return ONLY JSON: {{"home": <int>, "away": <int>, "timer": "<clock>"}}'
```

Needs `OPENAI_API_KEY` (or `GEMINI_API_KEY` with `--model gemini_ocr`) in `.env`.
429 rate limits back off automatically; per-detection resume makes re-runs free.

A full worked example (1k-frame run): **[runbooks/ipbl-1k-locateanything](runbooks/ipbl-1k-locateanything/README.md)**.
For a quick local loop instead, see the `prelabel` / `import-ls` (non-cloud) commands and `scripts/spike.py`.

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
  segment · transcribe   enrichment stages: boxes -> masks (SAM2) / crops -> text (OCR)
  server/              our FastAPI model-server: app · contract · adapters/{stub,owlv2,locateanything,sam2}
  adapters/{label_studio,coco} · ingest · verify · cli · config
  web/                 FastAPI app + static SPA (labeling-t-web)
scripts/             spike.py · ls_setup.py · serve_vllm.sh
runbooks/            worked end-to-end runbooks
tests/               157 tests
Dockerfile           the model-server image (the [models] extra; runs on the GPU pod)
docker-compose.yml + nginx/   local Label Studio + CORS image server
```

Console commands: `labeling-t` (prelabel[-cloud] / segment-cloud / transcribe[-cloud] /
import-ls[-cloud] / from-ls[-cloud] / to-coco / manifest / frames / ingest-images /
ls-config), `labeling-t-runpod`
(up / down / status / gpus / datacenters), `labeling-t-models` (the model-server,
runs on the pod), and `labeling-t-web` (browser console).
