# CLAUDE.md — working state & operational context

Read this first to resume efficiently. **Architecture:** [doc.md](doc.md). **Roadmap:** [plans/roadmap.md](plans/roadmap.md).
This file is the *current state* + operational facts that aren't obvious from the code.

## What this project is (1 line)
Batch auto-labeling backend: run a vision model over frames → neutral-schema labels →
verify in Label Studio → export for training. Owned neutral schema is the contract;
models / LS / COCO are swappable adapters. Branch: **`main`** (transformers-model-server
merged 2026-08-04 — sam3, workforce, keypoint pull-back all landed).

## Models & serving (in code, `models.py` registry)
Three backends behind one neutral schema:
- **transformers** (our FastAPI model-server, one GHCR image `ghcr.io/qvuer7/labeling-t-models`,
  `MODEL` env selects adapter): `owlv2`, `locate_anything`, `sam2`, `vitpose` (keypoints),
  `sam3` (text→masks+boxes+scores in one pass; `:sam3` image variant — transformers 5.14.1,
  built+pushed, GPU-TESTED on 5090 2026-07-20: 1488 frames @ ~0.5s/frame. Gated weights →
  HF_TOKEN in .env (access granted to az1029; gate is MANUAL Meta approval). Prompt lesson:
  direct concept names beat descriptions ("center circle of the basketball court" + --min-score
  0.8 clean vs arcs; "red rectangle painted on the basketball court floor" → paint 0.96+).
  Gotcha: `datacenters` cmd (runpodctl data) falsely reported 5090 out of stock — RunPod
  GraphQL gpuTypes said stock=High; trust GraphQL, fix datacenters someday.
- **vllm** (stock image): `qwen3_vl`.
- **hosted chat** (`ChatClient`, no GPU): `openai_vl`, `gemini_vl`.
Provision GPUs: `labeling-t-runpod up --model <k> --gpu <preset> --hours H --budget $`
(auto-retries datacenters with stock; refuses duplicates — `--force` overrides).
Provision HUMANS: `labeling-t-workforce` (workforce.py, RentAHuman backend, added 2026-07-20) —
search/post-bounty/status/applications/message; RENTAHUMAN_API_KEY in .env; money actions
(escrow/accept/release) deliberately NOT wrapped — web UI or `rentahuman` MCP (.mcp.json) only.
Live bounty: QC9eap1xrZJwUkVo8PkF ($25, 100-frame keypoint trial for LS project 20, funded+open).
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
- `labels-ocr-clean/all/` — **2000 files with scoreboard OCR** on `Detection.text` as JSON
  `{"home","away","timer"}` + `corrections.json` (human-verified fixes 2026-07-08). The raw
  `labels-ocr` set was deleted 2026-07-20 (superseded by this corrected version).
- `labels-hoop/all/` — 1159 hoop labels (LS project 13 source).
- `labels-yolo-sam2/all/` — 1054 model masks; source of LS project 11 predictions. KEEP until
  project 11 verification finishes (accepted-as-is exports copy from here), then delete.
- `verified/all/` — 457 old human BOXES (pre-mask era). Kept pending user decision — human work,
  not derivable; delete when confirmed obsolete.
- `manifest.json` rebuilt 2026-07-06 (5 categories; counts only standard prefixes, so named
  label sets don't appear in its totals).
Deleted 2026-07-06 (all verified byte-identical-in-combined or superseded): `verified-reviewed-clean`,
`labels-filtered`, `pre-filtered-verified`, `labels-yolo-seg`. Deleted 2026-07-20: `labels-rim`,
`labels-rim-verified`, `verified-rim` (rim masks verified merged into `ipbl-basketball-seg/yolo`,
class 1; the 10 non-merged rim frames are all in its `dropped.txt`) + `labels-ocr` — don't look for them.

**`weights/`** (top-level, moved from `models/` 2026-07-20): `weights/ipbl-basketball-seg/` = two seg
runs `ipbl_seg_yolo26l_1280/` + `ipbl_seg_yolo26x_1280/` (best.pt 63/142 MB + training artifacts).
Still under `models/`: `ipbl-scoreboard-ocr/` (89 MB), `ipbl-shot-vision/` — not yet moved.

**`datasets/ipbl-basketball-seg/`** — `labels/` (1488 flat, group "" — combined minus 43 dropped,
LS project 16 source) + `frames/all/` (1488, copied from yolo/images 2026-07-20 for framework
addressing) + `labels-v2/all/` (1488 = labels ∪ sam3 court zones; LS project 19 source) +
`labels-sam3-court-final/all/` (1488, zones only: paint 481 + center_circle 1158; sample sets
`labels-sam3-court{,2,3}`/`labels-sam3-ccab` = prompt-iteration leftovers, deletable) + `yolo/`
(253 MB Ultralytics export, classes ball/rim/player/referee/scoreboard) + `dropped.txt` (43).

Also `datasets/ipbl-scoreboard-kp/` (200 frames + seeded keypoint labels, 5 games, for
LS project 17), `datasets/ipbl-court-kp/` (same frames, empty court labels, LS project 18),
`datasets/ipbl-scoreboard-ocr/` (47 MB, type1-3 crop archives), `datasets/ipbl-shot-vision/`
(248 MB — traces/ + npz/csv labels) and `streams/` = 30 matches, ~15 GB (KEEP — source of truth).
`datasets/ipbl-basketball/` parent-pool frames DELETED 2026-07-08 (re-derivable from streams;
manifest.json kept). `eval/`: `scored-missed*.csv` (shot-clip labels; the 201 clips themselves
deleted — 689 clips labeled locally 2026-07-14 across 6 batches: 386 make / 268 miss / 30 unclear
/ 5 not_a_shot), `traces/` (134 MB, scored-missed eval), `possession/`, `vision_label_new/`.
`demos/possession/` (218 MB) kept for now. Bucket total after 2026-07-20 cleanup: ~19 GB.

## Label Studio
Hosted at `LS_URL` (`.env`), token `LS_API_KEY`. Login `admin@labeling-t.local` / droplet password.
Relevant projects: **36 = "farfem-v1 pseudo-seg verify (brush)"** (2186 tasks; yolo26x_1280
pseudo-labels from `training/ipbl-basketball-seg/farfem-v1` [ipbl agent's export, PRO
far/female venues] converted via `scripts/yolo_seg_to_labelset.py` THROWAWAY into
`datasets/ipbl-basketball-seg/{frames,labels-farfem-v1}/farfem/`; created 2026-08-04;
pull back: `from-ls-cloud --dataset ipbl-basketball-seg --group farfem --project-id 36
--name farfem-v1 --masks`) · **20 = "IPBL court keypoints — anchors v2"** (706 PRO-league court-visible
frames per `ipbl-basketball-seg/venue_types.csv`; 18 clicked anchors + 2 arc-sample labels,
exact config/instructions from vsoccer `court_reg_refs/labelstudio_project_instructions.md`;
empty set `labels-pro-court-anchors/all`; export = JSON-MIN, coords in PERCENT; created
2026-07-20. NOTE: venue_types.csv also maps 10 venues/3 leagues per frame — sam3 paint prompt
works ONLY on PRIME venues, all 7 PRO venues got 0 paint) ·
**19 = "ipbl-seg v2 + court zones (brush)"** (1488 tasks from
`ipbl-basketball-seg/labels-v2/all` = labels ∪ sam3 court zones [paint 481 / center_circle
1158, source="sam3", min-score 0.8]; created 2026-07-20 for zone verification; merge via
`scripts/merge_label_sets.py` THROWAWAY) · **18 = "ipbl court keypoints (29pt)"** (same 200 frames as 17, dataset
`ipbl-court-kp`; 29 index-prefixed court points, 0-4 mid / 5-16 near end / 17-28 far end
"_far"; 15/27+16/28 names INFERRED — arc_junction_right/arc_top; label only visible points) ·
**17 = "ipbl scoreboard keypoints (type2 mix)"** (200 frames from 5
games, seeded home/away/timer points at type-2 centers — drag to correct; dataset
`ipbl-scoreboard-kp`, created 2026-07-07; keypoint pull-back WORKS since 2026-07-28 —
`from-ls[-cloud] --keypoint-category <cat>` collapses a task's flat LS point regions into one
Detection with `keypoints` + enclosing bbox; empty-result annotations are skipped, no dims) ·
**16 = "ipbl-seg latest labels (brush)"** (1488 tasks from
datasets/ipbl-basketball-seg/labels — flat set, `--group ""`; created 2026-07-07) ·
**15 = "ipbl-1k rim masks (brush)"** (1125 SAM2 masks on verified rim boxes,
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
- On `transformers-model-server`: agent-interface PR-1..8 + skill + bootstrap docs,
  **merged to main and pushed 2026-07-07** (fast-forward, both branches at f38d79b+).
  232 tests pass. Live e2e acceptance run still pending (lifecycle.md sequence).
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
