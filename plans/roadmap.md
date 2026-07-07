# Labeling-T — Plan / Roadmap

Forward-looking companion to **[doc.md](doc.md)** (which describes what exists).
Ordered roughly by priority. The throughline stays the same: the neutral schema
is the contract; everything else is a swappable adapter.

---

## Done (recap)

- Neutral schema + geometry; owned on-disk labels.
- Model registry across **3 backends**: `transformers` (OWLv2, LocateAnything-3B,
  SAM2), `vllm` (Qwen3-VL), hosted chat (OpenAI, Gemini).
- Our transformers model-server: one image, `MODEL`-selected adapter, uniform
  `/infer`; `stub` adapter for CI.
- Two-stage detect→segment (detector boxes → SAM2 masks, COCO RLE on the wire).
- Cloud loop: `prelabel-cloud` (+ `--labels-name`), `import-ls-cloud`,
  `from-ls-cloud`, `to-coco`; S3/DO Spaces storage; dataset layout.
- RunPod provisioning with **datacenter auto-targeting**.
- Web console; 125 tests.

---

## Next (highest value first)

### 1. Persist masks end-to-end (close the SAM2 loop) — DONE except COCO export
- [x] Add `Detection.mask` (COCO RLE) to the neutral schema (`schema.py` recipe).
- [x] `segment-cloud` chains a label set's boxes → SAM2 over a whole dataset and
      writes masks in place (`segment.py`; per-detection resume, `--to-name` copies).
- [x] Label Studio config + import for masks (polygon and brush), pulled back
      by `from-ls-cloud`.
- [ ] **COCO export emits `segmentation`** (supervision already carries
      RLE/polygons) — the remaining gap; `to-coco` is boxes-only and local-only
      (REVIEW.md §4.2).

### 1b. OCR / region transcription — shipped (2026-07)
`transcribe[-cloud]`: crop matching regions → hosted VLM (openai_ocr/gemini_ocr
specs) → `Detection.text`; 429 backoff, `image_detail:low`, per-detection resume.
Follow-ups when needed:
- [ ] Label Studio per-region TextArea verification for `Detection.text`.
- [x] `from-ls-cloud --include-accepted --accepted-from <set>`: viewed-but-
      unsubmitted LS tasks pulled as verified (source prediction file copied
      byte-exact) — shipped 2026-07-07.

### 2. Make the transformers backend fast & safe under load
Today it's `--concurrency 1` (one model, one GPU, not reentrant).
- [ ] Server-side request **serialization** (a lock) so concurrency>1 is safe.
- [ ] True **batched** `/infer` (process N images per forward) where the model
      supports it — the real throughput lever.
- [ ] LocateAnything **single-pass multi-category**: parse the `<ref>label</ref>`
      tags from one generate instead of N per-category passes (≈Nx faster).

### 3. Widen GPU availability
- [ ] Rebuild the model image on **torch `cu128`** (drops the CUDA-13 host
      requirement; 5090s and most modern hosts deploy without resource-roulette).
      Lower the preset default container disk (60→~30) while at it.
- [ ] Optional `--data-center`/`--country` already exist; consider retry-across-DCs
      on transient "no resources".

### 4. Confidence routing (cut human load)
- [ ] Auto-accept high-confidence boxes; send only uncertain ones to Label Studio.
      (LocateAnything is scoreless — route by agreement/ensemble instead, or by a
      scored detector.)

---

## Later

- **Tracking** — `track_id` via `supervision` ByteTrack across video frames.
- ~~**Schema extension to keypoints**~~ — DONE 2026-07 (plans/keypoints-plan.md K1-K4:
  `Detection.keypoints`, VitPose adapter, `keypoints-cloud`, stats/render). Remaining:
  K5 (LS keypoint round-trip — pilot first) and K6 (COCO keypoints, inside the to-coco rework).
- **Multi-model A/B / ensemble** — registry + second pod; compare or vote.
- **`job.yaml`** — declarative run config (model, dataset, categories, category
  map, thresholds) to replace ad-hoc CLI args once runs recur.
- **RunPod API-native provisioning** — replace the `runpodctl` shell-out with the
  GraphQL API (better machine selection, no CLI dependency).
- **More export targets** — YOLO, FiftyOne (both ~free off `supervision`).
- **Grounding DINO** adapter — earmarked (`server/adapters/__init__.py`), parked
  in favor of LocateAnything/SAM2.

---

## Explicitly out of scope (for now)

- Training / fine-tuning models — this is a *labeling* backend; it produces COCO
  for a separate training pipeline.
- A bespoke annotation UI — Label Studio is the verification adapter.
- Realtime / streaming inference — batch pre-labeling is the job.
