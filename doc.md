# Labeling-T — Architecture & Status

Auto-labeling backend: run a vision model over raw images once, get pre-labels,
verify the uncertain ones by hand, export to a training format. The model carries
most of the labeling load; a human only corrects.

Status: **v0.x — boxes + masks + region OCR, images only. Pipeline proven
end-to-end on real basketball footage. 157 tests passing.**

Scope has grown from "one detector" to **a model registry across three serving
backends + a two-stage detect→segment pipeline**, but the contract below is
unchanged — that's the design working.

---

## 1. The one design principle

There is exactly one idea holding this project together:

> An **owned, neutral label schema** is the canonical representation. Every model,
> every serving backend, every export format is a **swappable adapter** hanging
> off it. Nothing else is allowed to become the contract.

This is a direct response to prior attempts that died from coupling everything to
one tool/model. Here the models, the verification UI (Label Studio), and the
export format (COCO) are all interchangeable. Swapping any of them touches one
adapter, not the pipeline.

```
   model adapters         ┌──────────────────────────┐    export adapters
 (3 serving backends)     │   NEUTRAL SCHEMA (ours)   │
 OWLv2 / LocateAnything ─►│  ImageLabels / Detection  │─► COCO (supervision)
 Qwen3-VL (vLLM)         ►│  abs-pixel xyxy boxes     │─► Label Studio import
 GPT-4o / Gemini (API)   ►│  category / score / source│─► (YOLO / FiftyOne, later)
 SAM2 (masks, stage 2)  ─►└──────────────────────────┘
                                      ▲
                          Label Studio verified labels flow back IN
```

The on-disk labels are *ours* (`labels/<frame>.json`), not any model's raw output
and not Label Studio's format. That decoupling is the whole point.

---

## 2. Pipeline (end to end)

```
 raw frames        prelabel (a registered model)        human verify        export
┌───────────┐  →  ┌────────────────────────────┐  pre-  ┌──────────────┐ back ┌────────┐
│ S3 / disk │     │ detector → neutral-schema   │ labels │ Label Studio │ ───► │  COCO  │
│  (.jpg)   │ ──► │   JSON  (+ SAM2 masks, opt) │ ─────► │ (correct it) │from- │ dataset│
└───────────┘     └────────────────────────────┘        └──────────────┘  ls  └────────┘
     ffmpeg            prelabel / prelabel-cloud         import-ls-cloud / from-ls-cloud
```

Per frame, prelabel: reads dims, runs the model, converts coords to absolute
pixels, maps model labels → your categories, filters by score, wraps in the
neutral schema, writes `labels/<frame>.json`.

**Enrichment stages (optional)** rewrite an existing label set in place (or to
a `--to-name` copy), resuming per detection so re-runs are free:
- `segment-cloud` — a detector's boxes → SAM2 box prompts → `Detection.mask`
  (COCO RLE), one mask per box (`segment.py`).
- `transcribe[-cloud]` — crop each matching region → hosted VLM (OpenAI/Gemini)
  reads it → `Detection.text` (`transcribe.py`). The OCR result stays attached
  to the region *within the whole frame's* labels; the crop is just transport.

---

## 3. Models & serving backends

A `ModelSpec` (in `models.py`) bundles everything **intrinsic** to a model —
served name, prompt, coord space, default categories, which backend hosts it.
Only the endpoint + key come from the environment, per model. **Three backends,
two transport clients:**

| Model (key) | Role | Backend | Where it runs |
|---|---|---|---|
| `owlv2` | open-vocab detector | `transformers` | our model-server (GPU pod) |
| `locate_anything` | grounding VLM detector | `transformers` | our model-server (GPU pod) |
| `sam2` | **segmenter** (box→mask) | `transformers` | our model-server (GPU pod) |
| `qwen3_vl` | grounding VLM detector | `vllm` | stock `vllm/vllm-openai` (GPU pod) |
| `openai_vl` | general VLM (GPT-4o) | `openai` | hosted API (no GPU) |
| `gemini_vl` | general VLM (Gemini) | `gemini` | hosted API (no GPU) |

- **`ChatClient`** drives every OpenAI-compatible chat endpoint — a vLLM box we
  rent *and* hosted OpenAI/Gemini. Text out → `spec.parse` → boxes. The three
  differ only by endpoint + a couple of quirk params (`extra_body`), carried on
  the spec.
- **`TransformersClient`** drives our model-server's structured `/infer` (boxes
  already abs-px, dims included) and its `segment()` (box prompts → masks).
- `client_for(spec)` picks the transport by backend; `prelabel` duck-types
  `infer` vs `infer_raw`, so orchestration is backend-agnostic.

Adding a model = a new `ModelSpec` (+ for the transformers backend, one server
adapter). The neutral schema and the pipeline don't move.

---

## 4. The transformers model-server (`server/`)

Our own FastAPI server, **one Docker image for every transformers model** —
`MODEL` (env) selects which adapter the pod loads at startup.

```
POST /infer  {image_url, queries[], params{}}  →  {width, height, detections[]}
GET  /health                                   →  {status, model, ready}
```

- **One wire contract** (`contract.py`): `InferRequest` / `InferResponse` /
  `WireDetection {bbox abs-px, label, score, mask?}`. `bbox` is original-image
  absolute pixels; `mask` is COCO RLE (only the segmenter fills it).
- **One adapter per model** (`server/adapters/`): `owlv2`, `locateanything`,
  `sam2`, plus `stub` (no torch, for CI/seam tests, `MODEL=stub`). Each adapter
  owns the model-specific input/post-processing; torch is imported lazily inside
  `load()`/`detect()` so importing the registry stays GPU-free.
- **Box prompts for SAM2 ride in `params`** (`boxes`/`labels`/`scores`) — no new
  payload type; detectors ignore them, SAM2 uses them.

Per-model gotchas worth knowing (documented inline in each adapter):
- **OWLv2** pads images to a square; boxes are post-processed against the square
  size then unpadded/clamped, dropping phantom boxes in the padding.
- **LocateAnything-3B** is a generative VLM with vendored model code that pins
  **transformers==4.57.1** (a guarded shim covers a future bump). It's queried
  **one category per pass** (correct labels, no confidence scores), so a batch
  run is several `generate()`s per frame → use `--concurrency 1` (see §6).
- **SAM2** uses transformers' **native** `Sam2Model` — plain torch, **no custom
  CUDA `_C` extension** — so it shares the slim image (not `facebookresearch/sam2`).

---

## 5. The neutral schema (the contract)

```python
class BBox(BaseModel):          # absolute pixels, xyxy
    x1: float; y1: float; x2: float; y2: float

class Detection(BaseModel):
    bbox: BBox
    category: str
    score: float | None         # model confidence; None once human-verified or generative
    source: str | None          # provenance, e.g. "locate-anything-3b"

class ImageLabels(BaseModel):
    image_path: str
    width: int; height: int      # REQUIRED — every coord conversion needs them
    detections: list[Detection]
```

Canonical convention: **absolute-pixel `xyxy`** (aligns with `supervision`, so
COCO export is near-free). `extra="forbid"` keeps the schema from accumulating
per-case junk.

Two horizontal extensions have since landed on `Detection`, both following the
recipe documented inline in `schema.py` (optional, box-derived fields):
`mask` (COCO RLE, from box-prompted SAM2) and `text` (region transcription from
the OCR stage; `None` = never attempted, `""` = attempted, nothing legible).
`ImageLabels` also carries `schema_version` (default `"1"`; absent in
pre-versioning files, written on every dump) so on-disk labels self-identify.
Still open: COCO export emits boxes only — `segmentation` from masks is
plans/roadmap.md territory.

---

## 6. Cloud loop (codified in `layout.py`)

Dataset-grouped; everything for one labeling effort is self-contained under its
dataset name in the bucket:

```
streams/<game>/<game>_NNN.ts                         # RAW video (untouched)
datasets/<dataset>/
    frames/<group>/<stem>.jpg                        # keyframes
    labels/<group>/<stem>.json                       # model pre-labels (neutral schema)
    labels-<name>/<group>/<stem>.json                # a 2nd model's pre-labels (namespaced)
    verified/<group>/<stem>.json                     # human-verified
    export/<version>/annotations.coco.json           # exports
```

Frame, pre-label, and verified label share the same stem, so they join by name —
no manifest needed. **`--labels-name`** namespaces a second model's pre-labels
into `labels-<name>/` so several models coexist without clobbering (e.g.
LocateAnything in `labels-locateanything/` while Qwen stays in `labels/`).

**Set selectors** (`DatasetLayout.set_prefix`): commands that point at an
existing label set take the storage leaf name as a neutral selector —
`labels`, `labels-<name>`, `verified`, `verified-<name>` — exactly what a
bucket listing shows under the dataset root. A typo'd selector is a loud
ValueError, not an empty scan. `manifest.json` indexes every set:
`namespaces` (per-set per-group counts) + `namespace_totals` + `generated_at`
alongside the legacy `groups`/`totals` (which cover only frames + the two
default sets).

Cloud commands (`labeling-t …`): `prelabel-cloud` (presigned frame URL → model →
labels in S3), `import-ls-cloud` (tasks carry presigned S3 URLs + pre-annotations),
`from-ls-cloud` (verified labels back to S3), `to-coco`.

> **Concurrency note:** the transformers model-server serves **one model on one
> GPU** and is *not* safe under concurrent `generate()` (unlike vLLM, which
> continuous-batches). Use `--concurrency 1` for the transformers backend; vLLM
> models can fan out.

---

## 7. Serving infrastructure (RunPod)

`labeling-t-runpod` provisions the GPU. Hardware comes from a `PodSpec` (`gpu.py`:
rtx3090/4090/5090, a40, a100, h100), the model from a `ModelSpec`. One command
builds the serving recipe and records the endpoint in `.labeling-t/pods.json`
(runtime pod state, `podstate.py`; `.env` stays secrets-only — inference commands
resolve their endpoint as `--endpoint` flag > newest recorded pod for the model >
`{PREFIX}_ENDPOINT` env (deprecated) > the spec's baked-in SaaS default, and
`status` reconciles the state against the live pod list):

- **transformers backend** → our GHCR image (`ghcr.io/qvuer7/labeling-t-models`),
  `MODEL`/`HF_MODEL` via env, readiness on `/health` (true only after weights load).
- **vLLM backend** → stock `vllm/vllm-openai`, docker-args from the spec,
  readiness on `/v1/models`.
- **hosted backends** (openai/gemini) → no pod; the spec bakes the base URL,
  `.env` carries only the key.

**Datacenter auto-targeting:** `up` queries `runpodctl datacenter list`, finds the
DCs that actually have the requested GPU in stock, and passes `--data-center-ids`
— so scarce GPUs (e.g. RTX 5090) deploy like the web console instead of blind-
picking a full machine. `--data-center` forces a list; `datacenters --gpu <preset>`
inspects stock. Commands: `up / down / status / gpus / datacenters`.

**Guardrails:** `up` refuses to rent when a pod for the same model already runs
(deterministic name match; the error carries the existing pod's id/endpoint —
`--force` rents a deliberate second instance). `up --budget <$>` is a hard cost
cap: a known GPU price caps `--hours` pre-create, and the created pod's actual
`costPerHr` is re-checked — over budget deletes the pod immediately and the
error suggests corrected `--hours`.

The **image is one package, scoped deps**: base `import labeling_t` is torch-free;
the `[models]` extra (torch/transformers/scipy/fastapi/+SAM2/+LocateAnything deps,
**transformers pinned 4.57.1**) is the only extra installed on the pod. The pod
fetches `image_url` over HTTP and returns boxes/masks; S3/Label-Studio/COCO live
on the thin client.

---

## 8. Components (`labeling_t/`)

| Module | Responsibility |
|--------|----------------|
| `schema.py` | The owned contract: `BBox`, `Detection`, `ImageLabels`. Abs-pixel `xyxy`, `extra="forbid"`. |
| `geometry.py` | All coordinate math: normalized↔abs, abs↔percent (LS), abs→COCO `xywh`. Isolated, heavily tested. |
| `models.py` | `ModelSpec` registry — model identity + which backend hosts it. WHAT to serve. |
| `model_client.py` | `ChatClient` (OpenAI-compat: vLLM/OpenAI/Gemini) + `TransformersClient` (`/infer` + `segment`); `client_for`. |
| `server/` | FastAPI model-server: `app`, `contract` (wire shape), `adapters/{stub,owlv2,locateanything,sam2}`. |
| `prelabel.py` | `parse_boxes` + batch orchestration (local & cloud), backend-agnostic, resume, failure manifest. |
| `segment.py` | Enrichment: a label set's boxes → SAM2 box prompts → `Detection.mask`, in place. Per-detection resume. |
| `transcribe.py` | Enrichment: crop matching regions → hosted VLM → `Detection.text` (OCR). Per-detection resume. |
| `storage.py` | `Storage`: local + S3 (DO Spaces). presigned URLs, ranged-dim reads. The cloud source/sink. |
| `layout.py` | `DatasetLayout` — the one definition of the bucket folder structure (+ `--labels-name`). |
| `frames.py` | Video → keyframes (ffmpeg) → storage. |
| `gpu.py` / `runpod.py` | GPU presets (WHERE) / RunPod provisioning + datacenter selection. |
| `adapters/label_studio.py` · `adapters/coco.py` | Neutral ↔ LS (import + pull-back) · neutral → COCO via `supervision`. |
| `cli.py` · `config.py` · `web/` | CLI entry · `.env` loading · FastAPI browser console over the pipeline. |
| `output.py` | Agent output mode: `--json` on every subcommand of both CLIs — one `{"ok", "result"/"error"}` envelope on stdout, prose to stderr, `ok` ⇔ exit code 0. |

---

## 9. Known limitations (current)

- **COCO export is boxes-only** — `Detection.mask` persists and round-trips
  through Label Studio, but `to-coco` does not emit `segmentation` yet, and it
  requires images on local disk (can't read `s3://` label paths). The export
  stage is the weakest link in the cloud loop (see REVIEW.md §4).
- **transformers backend is single-request** — one model, one GPU, `--concurrency 1`.
  No server-side batching/queue yet.
- **LocateAnything is per-category** (N `generate()`s per frame) — correct labels,
  but slower than a single multi-category pass.
- **Image pinned to torch `cu130`** (needs a CUDA-13 host) — narrows GPU
  availability; a `cu128` build would widen it (plans/roadmap.md).
- Single model per pod; multi-model A/B is a registry + second pod.
- GPU provisioning is `runpodctl` (no API-native module yet).

---

## 10. Status snapshot

- [x] Neutral schema + geometry (fully tested)
- [x] Model registry across 3 backends: `transformers` (OWLv2 / LocateAnything-3B
      / SAM2), `vllm` (Qwen3-VL), hosted chat (OpenAI / Gemini)
- [x] Our transformers model-server (one image, `MODEL`-selected adapter, uniform `/infer`)
- [x] Two-stage detect→segment codified: `segment-cloud` (boxes → SAM2 → `Detection.mask`, in place)
- [x] Region OCR: `transcribe[-cloud]` (crops → hosted VLM → `Detection.text`), 429 backoff, per-detection resume
- [x] `schema_version` on `ImageLabels` — on-disk labels self-identify
- [x] Cloud loop: frames → prelabel-cloud (+ `--labels-name`) → import-ls-cloud → from-ls-cloud → to-coco
- [x] RunPod provisioning + **datacenter auto-targeting**
- [x] Label Studio import (auto-config + pre-annotations) and verified pull-back
- [x] Web console (FastAPI + SPA) over the same pipeline
- [x] 157 tests passing

Forward plan: see **[plans/roadmap.md](plans/roadmap.md)**.
