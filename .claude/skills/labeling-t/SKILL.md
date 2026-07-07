---
name: labeling-t
description: Drive the Labeling-T auto-labeling pipeline end-to-end (rent GPU, prelabel, segment, OCR, Label Studio verify, pull back, inspect) through its --json CLI. Use when operating the labeling-t / labeling-t-runpod CLIs, datasets under s3://ml-cv-data, RunPod model serving, or Label Studio projects for this repo.
---

# Operating Labeling-T as an agent

Two CLIs, one contract: `labeling-t-runpod` provisions GPU pods, `labeling-t`
runs the pipeline stages. Architecture: `doc.md`. Current dataset state:
`CLAUDE.md`.

## Session start (always)

1. `labeling-t-runpod status --json` — reconciles pod state: prunes dead pods
   from `.labeling-t/pods.json`, adopts running `labeling-t-*` pods, and shows
   `model` / `endpoint` / `terminate_after` per pod. Run it before assuming
   anything is (or isn't) serving.
2. `.env` is SECRETS ONLY (API keys, S3 creds, LS token). Never write it.
   Endpoints live in `.labeling-t/pods.json`, maintained by `up`/`down`/`status`.

## Output contract — never parse prose

Every subcommand of both CLIs takes `--json`:

- stdout carries EXACTLY one envelope: `{"ok": true, "result": {...}}` or
  `{"ok": false, "error": {"message": ..., <structured detail>}}`.
- `ok` mirrors the exit code (ok ⇔ rc 0). All prose goes to stderr.
- Long stages emit progress JSON lines on stderr:
  `{"event":"progress","stage":…,"done":n,"total":N,"elapsed_s":s}`
  (first item, then every 25 items or 5 s, always the final item).
  `--progress-file <path>` atomically rewrites the latest event for polling.
- Error payloads carry the recovery: e.g. a duplicate-pod refusal includes
  `error.existing.{id,endpoint,cost_per_hr}` — reuse that endpoint.

## Command map

Provisioning (`labeling-t-runpod`):
`gpus` · `datacenters --gpu <k>` (stock check) · `up --model <k> --gpu <k>
--hours H --budget $ [--force]` · `status` · `down [<pod-id>|--all]`.

Pipeline (`labeling-t`), in lifecycle order:
`ingest-images` / `frames` → `prelabel-cloud` → `segment-cloud` (boxes→SAM2
masks) / `transcribe-cloud` (crops→OCR text) → `import-ls-cloud` → human
verification in LS → `from-ls-cloud [--include-accepted --accepted-from <set>]`
→ `to-coco`.

Inspection (any time): `stats` · `validate` · `diff` · `render` · `manifest`.

Label sets are addressed by **selector** — the storage leaf name, exactly what
a bucket listing shows: `labels`, `labels-<name>`, `verified`,
`verified-<name>` (flags `--set`, `--a/--b`, `--frames-from`,
`--accepted-from`). Typos fail loudly.

## Checkpoint discipline (the difference between a good and bad run)

- **Sample first, then batch**: run any expensive stage on `--stems <5-10>`
  first, `render` the output, LOOK at the PNGs, then run the full set.
- **After every stage**: `stats --json` (counts, mask/text coverage, sources)
  and `render --sample 8 --out /tmp/render-<stage>` + look. Net-in-box,
  hallucinated boxes, wrong-region OCR are all caught visually, cheaply.
- **Before believing an in-place rewrite**: `diff --a <src> --b <dst>` —
  `changed` compares normalized detections (image_path rewrites don't count);
  read the `changed` stems before concluding the stage worked.
- **Never delete a label set** without `diff` showing `byte_identical` equal
  to the full file count against its replacement.
- `validate --json` must be rc 0 before any export or LS import.

## Guardrails (cost + platform)

- ALWAYS pass `--hours` (auto-terminate backstop) and `--budget` on `up`.
  Over-budget pods self-delete and the error suggests corrected `--hours`.
- `up` refuses a duplicate pod for the same model — reuse
  `error.existing.endpoint`; only `--force` for a deliberate second instance.
- transformers backend (owlv2, locate_anything, sam2) = **`--concurrency 1`**
  (one GPU model, not reentrant). vLLM and hosted APIs fan out fine.
- Full details + LS/OCR/RunPod gotchas: `references/guardrails.md`.

## References

- `references/lifecycle.md` — the full worked recipe, command by command,
  with the JSON fields to read at each step.
- `references/ocr-enrichment.md` — transcribe stages: specs, keys, rate
  limits, prompt escaping, resume semantics.
- `references/guardrails.md` — every known cost/API/LS gotcha in one place.
