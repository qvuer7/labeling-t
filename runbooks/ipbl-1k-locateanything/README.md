# Runbook — LocateAnything-3B over `ipbl-basketball-1k` → Label Studio

Pre-label the 1,000-frame `ipbl-basketball-1k` dataset with **LocateAnything-3B**
(5 categories incl. the new `scoreboard` + `hoop`), save neutral-schema labels to
S3 **without touching the existing Qwen labels**, then load them into Label Studio
for box verification.

Categories: `player, ball, referee, scoreboard, hoop`
(model is queried with descriptive phrases, mapped to clean names — see
[`category_map.json`](category_map.json)).

## 0. Prereqs

- `.env` has S3 creds (`AWS_*`, `S3_BUCKET=ml-cv-data`, …) and Label Studio
  (`LS_URL`, `LS_API_KEY`). Frames already live at
  `s3://ml-cv-data/datasets/ipbl-basketball-1k/frames/all/` (1000 `.jpg`).
- Model image `ghcr.io/qvuer7/labeling-t-models:latest` is pushed (includes the
  LocateAnything adapter; transformers pinned to 4.57.1).

## 1. Serve LocateAnything on a GPU

```bash
labeling-t-runpod up --model locate_anything --gpu a40 --hours 8
# writes LOCATE_ANYTHING_ENDPOINT to .env; waits until /health is ready.
# A40 (48GB) — used because RTX-4090 SECURE was out of stock; rtx4090/rtx3090/rtx5090
# also fit the 3B model. Pod is CUDA-13 (the image ships torch cu130).
```

## 2. Pre-label the 1,000 frames → S3 (non-destructive)

```bash
labeling-t prelabel-cloud \
  --dataset ipbl-basketball-1k --group all --model locate_anything \
  --labels-name locateanything \
  --categories "basketball player,basketball,referee,scoreboard,basketball hoop" \
  --category-map runbooks/ipbl-1k-locateanything/category_map.json \
  --concurrency 1
```

- `--labels-name locateanything` → writes to
  `datasets/ipbl-basketball-1k/labels-locateanything/all/`, leaving the existing
  Qwen labels at `labels/all/` **untouched**. (New `DatasetLayout.labels(group, name)`.)
- `--concurrency 1` is **required** for the transformers model-server: it serves
  one model on one GPU and is not safe under concurrent `generate()` (unlike vLLM,
  which continuous-batches). Throughput ≈ **8 s/frame** (5 categories, one
  generate each via Parallel Box Decoding) → ~**135 min** for 1000 frames.
- Resumable: re-running skips frames already written; per-frame errors land in
  `labels-locateanything/all/failures.jsonl` and don't stop the batch.

## 3. Import into Label Studio (frames via presigned S3 URLs)

```bash
labeling-t import-ls-cloud \
  --dataset ipbl-basketball-1k --group all \
  --labels-name locateanything \
  --project "ipbl-1k LocateAnything" \
  --categories "player,ball,referee,scoreboard,hoop"
# uses LS_URL / LS_API_KEY from .env; prints the LS project id.
```

Boxes load as **pre-annotations**; frames are served straight from S3 via
presigned URLs (7-day TTL). Open the project and correct boxes.

## 4. Pull verified labels back → S3

```bash
labeling-t from-ls-cloud \
  --dataset ipbl-basketball-1k --group all \
  --project-id <ID printed in step 3>
# writes human-verified neutral-schema labels to verified/all/
```

## Notes / gotchas

- **Qwen labels preserved.** They stay at `labels/all/` (1000 files); LocateAnything
  lives in the parallel `labels-locateanything/all/`. Nothing was overwritten.
- **GPU availability.** RTX-4090 SECURE can be out of stock; A40/RTX-5090/A100 all
  fit the 3B model. The image needs a CUDA-13 host (torch cu130).
- **LocateAnything has no confidence scores** (generative), so labels carry
  `score=null`; `--min-score` is a no-op for this model.
- **Tear down when done:** `labeling-t-runpod down <pod-id>` (stops billing).
```
