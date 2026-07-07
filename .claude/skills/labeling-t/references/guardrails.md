# Guardrails and gotchas — every known trap, one page

## Money (RunPod)

- ALWAYS `--hours` (auto-terminate backstop) + `--budget $` on `up`. A pod
  whose actual $/hr × hours exceeds the budget is deleted immediately; the
  error carries `suggested_hours`.
- Duplicate refusal: `up` fails when a pod for the same model runs;
  `error.existing.endpoint` is the one to reuse. `--force` = deliberate
  second instance (twice the billing).
- `down <id>` the moment a stage finishes; `down --all` + `status --json`
  (expect `pods: []`) before ending a session.
- A `not ready before timeout` failure still leaves a BILLING pod — the
  result payload has its id; decide (wait vs down) explicitly.
- 4090/5090 SECURE are often out of stock; `a40` and 3090-COMMUNITY are the
  reliable fallbacks. `datacenters --gpu <k>` shows live stock; `up` already
  retries stocked datacenters in order.

## Serving / inference

- transformers backend (`owlv2`, `locate_anything`, `sam2`) serves ONE model
  on ONE GPU and is not reentrant: **`--concurrency 1`** on prelabel-cloud /
  segment-cloud against it. vLLM (`qwen3_vl`) and hosted APIs fan out.
- Endpoint resolution: `--endpoint` flag > newest recorded pod for the model
  (`.labeling-t/pods.json`) > `{PREFIX}_ENDPOINT` env (deprecated) > the
  spec's SaaS default. Connection refused on a recorded pod ⇒ run
  `status --json` (reconciles state), then decide.
- Never write `.env`. It holds secrets only.

## Label Studio

- Project titles max **50 chars** (hosted LS 400s opaquely above that).
- Presigned frame URLs expire in ~7 days — if images stop loading in LS,
  re-run `import-ls-cloud` (same project title makes a NEW project; pull the
  old one first).
- Default export = ANNOTATED tasks only. Viewed-but-unsubmitted tasks need
  `from-ls-cloud --include-accepted --accepted-from <set>` (copies the source
  prediction file byte-exact). `result.missing_source` lists accepted stems
  whose source file vanished — investigate before trusting counts.
- Partially-verified projects sliced by task id stay in
  `scripts/export_prefiltered_verified.py` (throwaway) — the id rule is
  deliberately NOT a framework flag.

## Data safety

- Deletion rule: a label set may be deleted ONLY after
  `diff --a <candidate> --b <replacement>` shows `byte_identical` == file
  count (or every difference is understood and recorded).
- In-place enrichment (`segment-cloud`/`transcribe-cloud` without
  `--to-name`) rewrites the set. When unsure, `--to-name` a copy and `diff`.
- `manifest --json` after manual S3 surgery — `namespaces` /
  `namespace_totals` now count every named set; `generated_at` says how
  fresh it is.

## OCR

- `--prompt` is format()ed: double literal braces (`{{` `}}`).
- 429s back off via Retry-After; lower `--concurrency` before blaming the API.
- `Detection.text`: None = never attempted, "" = attempted-illegible; resume
  skips non-None. Don't "clean" empty strings to null — that re-buys crops.

## Output discipline

- Parse ONLY the stdout JSON envelope; prose and progress live on stderr and
  may change wording at any time.
- `ok` ⇔ exit code 0. Error envelopes carry structured recovery fields —
  read them before retrying blind.
