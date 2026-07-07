# The full lifecycle, worked — zero prose parsing, zero .env edits

Every step names the JSON fields to read. Substitute your dataset/group/model.
This is also the acceptance sequence the framework is tested against.

## 0. Preflight

```bash
labeling-t-runpod status --json
# read: result.balance, result.pods[] (model/endpoint/terminate_after/ready),
#       result.stale_removed (dead pods pruned from state)
labeling-t-runpod datacenters --gpu a40 --json
# read: result.datacenters[] — empty/error means no stock; try another --gpu
```

## 1. Serve a model (GPU-backed stages only; hosted OCR needs no pod)

```bash
labeling-t-runpod up --model locate_anything --gpu a40 --hours 3 --budget 3 --json
# rc 0: read result.id, result.endpoint, result.ready, result.terminate_after
# rc 1 duplicate: read error.existing.endpoint — REUSE it, don't rent again
# rc 1 over budget: read error.suggested_hours — re-run with corrected --hours
# rc 1 not-ready timeout: pod is up and billing; result carries id/endpoint —
#   check again later or `down <id>`
```

Inference commands resolve the endpoint automatically (newest recorded pod for
the model). `--endpoint <url>` overrides when juggling several pods.

## 2. Ingest frames

```bash
labeling-t ingest-images --src ./my-frames --dataset my-ds --group all --json
# read: result.uploaded, result.total, result.skipped (resume), result.dest
```

## 3. Prelabel — sample first

```bash
labeling-t prelabel-cloud --dataset my-ds --group all --model locate_anything \
    --categories cat,dog --concurrency 1 --stems f001,f002,f003,f004,f005 --json
# read: result.requested, result.matched, result.labeled, result.failures_file
labeling-t render --dataset my-ds --group all --set labels --out /tmp/r1 --json
# LOOK at /tmp/r1/*.png before committing GPU time to the full set
```

Then the full run (drop `--stems`; progress arrives on stderr):

```bash
labeling-t prelabel-cloud --dataset my-ds --group all --model locate_anything \
    --categories cat,dog --concurrency 1 --json
labeling-t stats --dataset my-ds --group all --set labels --json
# read: result.files, result.detections, result.by_category, result.sources
labeling-t validate --dataset my-ds --group all --set labels --json   # must be rc 0
```

## 4. Enrich: masks (and/or OCR — see ocr-enrichment.md)

```bash
labeling-t-runpod down <detector-pod-id> --json     # stop paying for the detector
labeling-t-runpod up --model sam2 --gpu a40 --hours 2 --budget 2 --json
labeling-t segment-cloud --dataset my-ds --group all --concurrency 1 --json
# read: result.segmented; then verify visually + numerically:
labeling-t stats --dataset my-ds --group all --set labels --json
# read: result.masks.coverage (expect ~1.0), result.masks.files_fully_masked
labeling-t render --dataset my-ds --group all --set labels --sample 8 --out /tmp/r2 --json
labeling-t-runpod down <sam2-pod-id> --json
```

In-place rewrite paranoia: `--to-name enriched` writes copies instead, then
`diff --a labels --b labels-enriched` shows exactly what changed.

## 5. Label Studio verification

```bash
labeling-t import-ls-cloud --dataset my-ds --group all --project "my-ds masks" \
    --categories cat,dog --masks --mask-format brush --json
# read: result.project_id, result.project_url  (title must be <= 50 chars)
```

Human verifies in LS (presigned URLs die in ~7 days — re-import if stale).
Then pull back, including tasks accepted by viewing without edits:

```bash
labeling-t from-ls-cloud --dataset my-ds --group all --project-id <id> \
    --name masks --include-accepted --accepted-from labels --json
# read: result.pulled, result.corrected, result.accepted, result.missing_source
labeling-t diff --dataset my-ds --group all --a labels --b verified-masks --json
# read: result.changed (what the human fixed), result.byte_identical (accepted copies)
```

## 6. Wrap up

```bash
labeling-t stats --dataset my-ds --group all --set verified-masks --json
labeling-t to-coco --labels <local-dir> --out annotations.coco.json --json   # boxes-only today
labeling-t-runpod down --all --json
labeling-t-runpod status --json          # confirm result.pods == []
```
