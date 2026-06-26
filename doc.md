# Labeling-T — Architecture & Status

Auto-labeling backend: run a foundation model over raw images once, get
pre-labels, verify the uncertain ones by hand, export to a training format.
The model carries most of the labeling load; a human only corrects.

Status: **v0 — boxes only, images only, one model. Pipeline proven end-to-end.**
61 tests passing.

---

## 1. The one design principle

There is exactly one idea holding this project together:

> An **owned, neutral label schema** is the canonical representation. Every model,
> every tool, every export format is a **swappable adapter** hanging off it.
> Nothing else is allowed to become the contract.

This is a direct response to prior attempts that died from over-generalizing and
coupling everything to one tool/model. Here, the model (Qwen3-VL), the
verification UI (Label Studio), and the export format (COCO) are all
interchangeable. Swapping any of them touches one adapter, not the pipeline.

```
                       ┌──────────────────────────┐
   model adapters  →   │   NEUTRAL SCHEMA (ours)   │   →  export adapters
                       │  ImageLabels / Detection  │
   Qwen3-VL ──────────►│  abs-pixel xyxy boxes     │──────► COCO (supervision)
   (LocateAnything)    │  category / score / source│──────► Label Studio import
   (SAM2, later)       └──────────────────────────┘──────► (YOLO / FiftyOne, later)
                                   ▲
                       Label Studio verified labels
                       flow back IN via from_label_studio
```

The on-disk labels are *ours* (`labels/<frame>.json`), not the model's raw output
and not Label Studio's format. That decoupling is the whole point.

---

## 2. Pipeline (end to end)

```
 raw frames                prelabel (remote model)            human verify            export
┌───────────┐   images    ┌────────────────────────┐   pre-   ┌──────────────┐  back  ┌──────────┐
│ S3 / disk │ ──────────► │ Qwen3-VL on vLLM (GPU)  │  labels  │ Label Studio │ ─────► │  COCO    │
│  (.jpg)   │             │  → parse → convert →    │ ───────► │  (correct    │ from-ls│ dataset  │
└───────────┘             │     neutral schema JSON │          │   the boxes) │        └──────────┘
                          └────────────────────────┘          └──────────────┘
        ffmpeg                  scripts/spike.py                import-ls / from-ls / to-coco
   (video → frames)            labeling_t.prelabel              labeling_t.adapters
```

Per frame, `prelabel`:
1. reads image dimensions (W/H),
2. POSTs image + prompt to the vLLM endpoint (`model_client.VLLMClient`),
3. parses the model's text → `(box, label, score)` (`prelabel.parse_boxes`),
4. converts coords to absolute pixels (`geometry`), clamps to image bounds,
5. maps model labels → your categories, filters by score,
6. wraps the result in the neutral schema and writes `labels/<frame>.json`.

---

## 3. Components (`labeling_t/`)

| Module | Responsibility |
|--------|----------------|
| `schema.py` | The owned contract: `BBox`, `Detection`, `ImageLabels`. Absolute-pixel `xyxy`. `extra="forbid"` so the schema can't sprawl. v0 is Detection-only; mask/keypoint/track_id extension recipe is documented inline. |
| `geometry.py` | All coordinate math in one place: normalized↔abs, abs↔percent (Label Studio), abs→COCO `xywh`. The single bug-prone zone, isolated and heavily tested. |
| `models.py` | `ModelSpec` registry. Each model's intrinsic behavior — served name, prompt, `coord_space`, default categories, parser — lives here, **not** in `.env`. |
| `gpu.py` | `PodSpec` + GPU preset registry (rtx4090/5090, a40, a100, h100…). Rentable hardware config, separate from model behavior. |
| `runpod.py` | RunPod provisioning (the `labeling-t-runpod` command): `up`/`down`/`status`/`gpus`. Builds the serving recipe from a `ModelSpec` + `PodSpec`, writes the endpoint to `.env`. |
| `model_client.py` | `VLLMClient`: httpx transport to a remote vLLM OpenAI-compatible endpoint. Sends image (base64) + prompt; caps `max_tokens`; applies `repetition_penalty`; retries 5xx/network. |
| `prelabel.py` | `parse_boxes` (truncation-tolerant, deduping) + batch orchestration (`prelabel`): bounded-concurrency requests, streams from disk, resumes by skipping already-written frames, routes per-frame errors to `failures.jsonl`. |
| `adapters/label_studio.py` | Neutral → LS tasks + pre-annotations (percent coords), auto-generated labeling config from categories, and `from_label_studio` to pull verified labels back. |
| `adapters/coco.py` | Neutral → COCO via Roboflow `supervision` (gives YOLO + tracking for free later). |
| `cli.py` | `labeling-t {prelabel, import-ls, from-ls, to-coco, ls-config}`. |
| `config.py` | Loads `.env` (infra only). |

`scripts/`: `spike.py` (per-model run/diagnostic), `ls_setup.py` (enable LS API
tokens), `serve_vllm.sh` (reference GPU serve command).

---

## 4. The neutral schema (the contract)

```python
class BBox(BaseModel):          # absolute pixels, xyxy
    x1: float; y1: float; x2: float; y2: float

class Detection(BaseModel):
    bbox: BBox
    category: str
    score: float | None         # model confidence; None once human-verified
    source: str | None          # provenance, e.g. "qwen3-vl"

class ImageLabels(BaseModel):
    image_path: str
    width: int; height: int      # REQUIRED — every coord conversion needs them
    detections: list[Detection]
```

Canonical convention: **absolute-pixel `xyxy`**. This aligns with `supervision`'s
internal representation (so COCO export is free) and makes every other system a
single conversion away. `extra="forbid"` rejects unknown fields, which is what
keeps the schema from accumulating per-case junk.

**Extension (documented in `schema.py`, do when a second type lands):** add a
shared `Annotation` base for the common fields, add sibling lists (`masks`,
`keypoints`) on `ImageLabels`. Nothing built ahead of need.

---

## 5. Model serving

The model runs on a **remote GPU**; this codebase has no torch and needs no GPU —
it only speaks HTTP to a vLLM OpenAI-compatible endpoint.

- **Active model: `Qwen3-VL-8B-Instruct`** (spec key `qwen3_vl`). Modern open
  grounding VLM. Outputs `[{"bbox_2d":[x1,y1,x2,y2],"label":..}]` in **0–1000
  normalized** coords (`coord_space="norm1000"`).
- **Dropped: `LocateAnything-3B`** — its custom architecture isn't servable on
  stock `vllm/vllm-openai` (kept in the registry for if/when a compatible image
  exists). The pivot was a one-spec change, which is the design working.
- **Serving (RunPod):** one command — `labeling-t-runpod up [--model qwen3_vl]
  [--gpu rtx4090|a100|...]`. It builds the vLLM docker-args from the `ModelSpec`,
  picks hardware from a `PodSpec`, pins a CUDA-13 node, sets an auto-terminate
  backstop, rents the GPU, and writes the endpoint to `.env`. `down` stops
  billing; `gpus` lists presets. (Recipe details + gotchas also in
  `scripts/serve_vllm.sh` and project memory.)
- **Two non-obvious knobs** (in `VLLMClient`): `max_tokens` cap and
  `repetition_penalty` — without them grounding VLMs loop the same box to the
  context limit (minutes/request, truncated JSON).

### Batch inference

Already handled. `prelabel` fires concurrent requests (`--concurrency`, a thread
pool); vLLM's **continuous batching** packs them onto the GPU. You scale
throughput by raising concurrency, not by manually batching. For huge datasets:
streaming-from-disk and per-frame-resume are built in.

---

## 6. Label Studio integration

`import-ls` creates a project, **auto-generates the labeling config** from your
categories (so config and labels can't drift), and uploads tasks with
pre-annotations (boxes converted to the percent coords LS expects). After a human
corrects boxes, `from-ls` pulls them back into the neutral schema, and `to-coco`
exports. That round-trip is the reason the backend owns the labels.

Solved along the way (baked into scripts/compose):
- **Auth:** LS 1.23 disables SDK (legacy) tokens by default → `scripts/ls_setup.py`
  enables them and returns the key.
- **Images:** LS's built-in local-file serving is tied to registered storage and
  kept 404-ing → a small **nginx sidecar** serves frames over HTTP with CORS
  (`nginx/images.conf`), and tasks reference normal `http://` URLs.

---

## 7. What's implemented

- [x] Neutral schema + geometry conversions (fully tested)
- [x] Remote vLLM client (retries, token cap, repetition penalty)
- [x] Batch prelabel orchestration (concurrency, streaming, resume, failure manifest)
- [x] Model registry / swappable `ModelSpec` (Qwen3-VL active)
- [x] Label Studio import with auto-config + pre-annotations
- [x] Verified-label pull-back (`from-ls`) and COCO export (`to-coco`)
- [x] CLI + per-model spike script
- [x] Local dev infra (docker-compose: LS + image server)
- [x] RunPod GPU provisioning (manual, via `runpodctl`)
- [x] Proven end-to-end on real basketball footage (12 frames, 4 games)

---

## 8. Local dev vs production (the real gap)

The current demo runs **images and Label Studio locally**. That is scaffolding,
not the intended deployment — and the architecture already separates the two so
the move is mostly config, not code:

| Concern | Local (now) | Production (intended) |
|---------|-------------|------------------------|
| **Frames** | `data/spike_frames/` on disk, served by an nginx sidecar | Live in **S3 / DO Spaces** (`ml-cv-data`). Tasks reference **presigned URLs**. |
| **Label Studio** | `docker compose` on `localhost:8080` | A **hosted/shared LS** instance. The scripts already take `--url`, so just point elsewhere. |
| **Model** | Already remote (rented GPU) | Same, or a persistent inference endpoint. |
| **Image URLs** | `--image-base-url http://localhost:8081` | `--image-base-url` → an S3 public/presigned base, or a small presigned-URL adapter. |

**What this needs to be production-real:**
1. An **S3 image-ref adapter** — instead of the nginx base URL, generate presigned
   S3 URLs for each frame (small addition to `_image_ref`). The frames already
   originate from the bucket, so the source of truth is S3, not local disk.
2. **Point `import-ls --url` at the hosted Label Studio.** No code change.
3. Optionally, **frames stay in S3 end-to-end**: read frames from S3 for
   inference (the model client already takes a URL or bytes), label, push S3 URLs
   to LS. Nothing is held locally.

In short: nothing about the *labeling logic* assumes local — it's the two infra
conveniences (nginx, dockerized LS) that are local, and both are swappable.

---

## 9. Known limitations (v0 scope)

- Boxes only. No masks, keypoints, tracking, or video (deferred by design).
- Single model at a time. Multi-model A/B is a registry addition.
- Ball recall is weak at 8B / 720p (players + referees are solid). A larger
  Qwen3-VL or a better prompt would help.
- GPU provisioning is manual `runpodctl`; no automated up/down module yet.
- Image serving for LS is local nginx (see section 8).

---

## 10. Roadmap (v1+)

- **S3 image serving** (presigned URLs) — removes all local image dependence.
- **Hosted Label Studio** target (config only).
- **SAM2 stage** — boxes from the VLM prompt SAM2 for masks (the two-stage
  `PreLabeler` design is already in place conceptually).
- **Tracking** (`track_id` via supervision ByteTrack) + video frame extraction.
- **Schema extension** to masks/keypoints (recipe already in `schema.py`).
- **Confidence routing** — auto-accept high-confidence boxes, only send uncertain
  ones to humans (cuts verification load).
- **Larger / multiple models** behind the registry.
- **`job.yaml`** declarative config (replaces ad-hoc CLI args) once jobs recur.
- **RunPod auto-provisioning** module (up → wire endpoint → run → teardown).

---

## 11. Quickstart

```bash
# install (no GPU / torch needed locally)
uv sync --extra integrations

# run tests
uv run pytest -q

# --- serve a model on a rented GPU (RunPod) ---
labeling-t-runpod gpus                     # list GPU presets
labeling-t-runpod up --model qwen3_vl --gpu rtx4090   # rent + serve + write .env
#   ... label ...
labeling-t-runpod down                     # stop billing

# --- label a folder of frames ---
uv run python scripts/spike.py --model qwen3_vl --images data/spike_frames --out labels
#   --check  : is the endpoint serving?
#   --raw    : dump raw model output (for tuning a new model's parser)

# --- human verification round (local dev) ---
docker compose up -d                       # Label Studio :8080 + image server :8081
uv run python scripts/ls_setup.py          # -> API token
labeling-t import-ls --labels labels --api-key <token> \
    --project basketball-spike --categories player,ball,referee \
    --image-base-url http://localhost:8081 --image-root data

# --- after correcting boxes in the UI ---
#   export from LS, then:
labeling-t from-ls --export export.json --out verified/
labeling-t to-coco --labels verified/ --out dataset.coco.json
```

Label Studio UI: http://localhost:8080 (`admin@labeling-t.local` / `labeling-t-admin`).

---

## 12. Repo layout

```
src/labeling_t/         # the framework (src-layout, pip-installable)
  schema.py             # the owned contract
  geometry.py           # coordinate conversions
  models.py             # ModelSpec registry (qwen3_vl active) — WHAT to serve
  gpu.py                # PodSpec GPU presets               — WHERE to serve
  runpod.py             # provisioning (labeling-t-runpod up/down/status/gpus)
  model_client.py       # remote vLLM transport
  prelabel.py           # parse + batch orchestration
  adapters/
    label_studio.py     # import + pull-back
    coco.py             # export via supervision
  cli.py                # labeling-t entry point
  config.py             # .env loading
scripts/                # thin operational entries (not core framework)
  spike.py              # per-model run / diagnostic
  ls_setup.py           # enable LS API tokens
  serve_vllm.sh         # reference GPU serve command
tests/                  # 66 tests
docker-compose.yml      # local LS + image server
nginx/images.conf       # CORS for the image server
.env / .env.example     # infra config (endpoints/keys) only

Console commands (installed): `labeling-t` (prelabel/import-ls/from-ls/to-coco/
ls-config) and `labeling-t-runpod` (up/down/status/gpus).
```
