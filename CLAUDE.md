# CLAUDE.md — working state & operational context

Read this first to resume efficiently. **Architecture:** [doc.md](doc.md). **Roadmap:** [plans/roadmap.md](plans/roadmap.md).
This file is the *current state* + operational facts that aren't obvious from the code.

## What this project is (1 line)
Batch auto-labeling backend: run a vision model over frames → neutral-schema labels →
verify in Label Studio → export for training. Owned neutral schema is the contract;
models / LS / COCO are swappable adapters. Branch: **`transformers-model-server`**.

## Models & serving (in code, `models.py` registry)
Three backends behind one neutral schema:
- **transformers** (our FastAPI model-server, one GHCR image `ghcr.io/qvuer7/labeling-t-models`,
  `MODEL` env selects adapter): `owlv2`, `locate_anything`, `sam2`.
- **vllm** (stock image): `qwen3_vl`.
- **hosted chat** (`ChatClient`, no GPU): `openai_vl`, `gemini_vl`.
Provision GPUs: `labeling-t-runpod up --model <k> --gpu <preset> --hours H --budget $`
(auto-retries datacenters with stock; refuses duplicates — `--force` overrides).
Endpoint state lives in `.labeling-t/pods.json` (podstate.py), NOT `.env` (secrets only);
inference commands auto-resolve it (`--endpoint` overrides); `status --json` reconciles.

### Gotchas that cost time (don't relearn)
- **transformers backend = `--concurrency 1`** (one GPU model, not safe under concurrent `generate`; vLLM is).
- **transformers pinned `==4.57.1`** in `[models]` — LocateAnything vendors a Qwen2 snapshot that breaks on 5.x.
- **SAM2 uses transformers-native `Sam2Model`** (plain torch, no CUDA `_C` compile) — same slim image.
- **RunPod:** 4090/5090 SECURE often out of stock; A40 / 3090-COMMUNITY are reliable fallbacks.
  `labeling-t-runpod datacenters --gpu <k>` shows stock. runpodctl honors only ONE datacenter per create.
- **Masks** ride on `Detection.mask` as **COCO RLE**; LS import/pull-back does polygon or brush
  (`import-ls-cloud --masks --mask-format {polygon,brush}`, `from-ls-cloud --name <ns>`).
- **Enrichment stages** (rewrite a label set in place, per-detection resume, `--to-name` for copies):
  `segment-cloud` boxes→SAM2→`Detection.mask`; `transcribe[-cloud]` crops→hosted VLM→`Detection.text`
  (OCR; specs `openai_ocr`/`gemini_ocr`, keys `OPENAI_API_KEY`/`GEMINI_API_KEY`).
- **OpenAI rate limits**: 429s honored via Retry-After (15/30s default); `image_detail="low"` on
  OCR specs cuts image tokens ~3x. Full 2k OCR pass ≈ $1.30 total.
- **LS export**: default export = ANNOTATED tasks only; `from-ls-cloud --include-accepted
  --accepted-from <set>` pulls viewed-but-unsubmitted tasks too (byte-exact source copy).
  Only id-threshold slicing of a PARTIALLY verified project still needs
  `scripts/export_prefiltered_verified.py` (THROWAWAY).
- **`ImageLabels.schema_version`** = "1" (absent in pre-2026-07-02 files; they load fine).

## Cloud state — `s3://ml-cv-data` (DigitalOcean Spaces, creds in `.env`, gitignored)
Dataset **`datasets/ipbl-basketball-1k/`** (2000 frames, 28 matches). CLEANED 2026-07-06; holds ONLY:
- `frames/` — 2000 images (note: re-uploaded at higher res at some point)
- `labels-combined/all/` — **1532 mask labels = THE training set** = old verified-reviewed-clean
  (946 human) ∪ labels-filtered (586 = LS-project-11 verified 600 minus 14 user-dropped).
  13,006 detections, 100% masked; provenance on `source`: 10,011 human / 2,995 best.pt+sam2.
- `labels-ocr/all/` — **2000 files with scoreboard OCR** on `Detection.text` as JSON
  `{"home","away","timer"}` (1987 parsed clean; ~22 score-decrease anomalies worth human eyes).
- `labels-yolo-sam2/all/` — 1054 model masks; source of LS project 11 predictions. KEEP until
  project 11 verification finishes (accepted-as-is exports copy from here), then delete.
- `verified/all/` — 457 old human BOXES (pre-mask era). Kept pending user decision — human work,
  not derivable; delete when confirmed obsolete.
- `models/ipbl-basketball-seg/primary/` — trained seg model `best.pt` (63 MB).
- `manifest.json` rebuilt 2026-07-06 (5 categories; counts only standard prefixes, so named
  label sets don't appear in its totals).
Deleted 2026-07-06 (all verified byte-identical-in-combined or superseded): `verified-reviewed-clean`,
`labels-filtered`, `pre-filtered-verified`, `labels-yolo-seg` — don't look for them.

Also `datasets/ipbl-basketball/` = raw parent pool (13,265 frames, 28 groups, no labels) and
`streams/` = 147 raw videos (12.9 GB).

## Label Studio
Hosted at `LS_URL` (`.env`), token `LS_API_KEY`. Login `admin@labeling-t.local` / droplet password.
Relevant projects: **15 = "ipbl-1k rim masks (brush)"** (1125 SAM2 masks on verified rim boxes,
imported 2026-07-06 — mask-quality review) · **14 = "ipbl-1k rim boxes (verify before SAM2)"**
(1159; VERIFIED 1125, pulled to verified-rim/ + segmented → labels-rim-verified/; done) ·
**13 = "ipbl-1k hoop frames (brush)"** (1159) ·
**12 = "ipbl-1k combined training set (brush)"** (1532) · **11 = "ipbl-1k YOLO+SAM2 masks
(brush, unverified)"** (1054; ids < 12166 verified per user rule) · 9, 8 = superseded/old.
Presigned frame URLs expire in ~7 days — re-run import if links die.
Gotcha: LS project titles max **50 chars** (400 Validation error above that).

## ACTIVE WORK (2026-07-07)
1. **Agent-interface plan: PR-0..PR-8 ALL IMPLEMENTED** (plans/agent-interface-plan.md) —
   podstate + guardrails + selectors + stats/validate/diff + subsetting/progress + render +
   --include-accepted + skill (`.claude/skills/labeling-t/`). 232 tests green.
   NEXT: user runs the live e2e acceptance sequence (lifecycle.md) on a small dataset.
2. **Mask verification** — LS project 11 ongoing; when done: `from-ls-cloud
   --include-accepted --accepted-from labels-yolo-sam2`; partial slices still via
   `scripts/export_prefiltered_verified.py` (id threshold).
3. **OCR** — full 2k scoreboard pass done (`labels-ocr/`). Next per plan: temporal-consistency
   flagged frames (~22 score decreases) → human check → trusted eval set; then synthetic
   font-render training data for a small OCR model (out of framework scope — scripts).
4. **Export gap** — `to-coco` is boxes-only + local-only; masks can't reach a training format
   through the framework yet (REVIEW.md §4.2, plans/roadmap.md §1 remaining item).

Scoreboard crop-region scripts (`scripts/crop_boxes.py`, `crop_relative.py`, tuned boxes for
3 scoreboard layouts) are dormant — superseded by whole-scoreboard OCR via `transcribe-cloud`,
still useful if per-digit-region OCR (Option B) is ever needed. Local data gitignored under
`data/` (match_samples = 28 matches × 10 frames + mask JSONs).

## Local / git
- On `transformers-model-server`: agent-interface PR-1..8 committed 2026-07-07 (7 commits,
  a213004..HEAD), NOT yet merged/pushed — pending user's live e2e test. 232 tests pass.
- `data/` and `.labeling-t/` are gitignored (local frames/masks/crops + pod runtime state).

## Common commands
```bash
uv run pytest -q                                   # 232 tests
labeling-t-runpod status --json                    # session start: pods + state reconcile
labeling-t stats --dataset ipbl-basketball-1k --group all --set labels-combined --json
labeling-t render --dataset ipbl-basketball-1k --group all --set labels-combined \
    --sample 8 --out /tmp/render --json            # LOOK at labels
labeling-t-runpod datacenters --gpu a40            # check GPU stock
labeling-t-runpod up --model sam2 --gpu a40        # rent+serve; down <id> to stop billing
# S3 (aws cli with DO Spaces endpoint):
aws s3 sync s3://ml-cv-data/datasets/ipbl-basketball-1k/labels-combined/all/ ./masks \
    --endpoint-url https://fra1.digitaloceanspaces.com
```
