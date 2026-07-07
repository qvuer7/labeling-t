# OCR / transcription enrichment (`transcribe`, `transcribe-cloud`)

Second-stage enrichment: crop each matching detection, send the crop to a
hosted VLM, write the reading onto `Detection.text`. No GPU pod — the OCR
specs are hosted APIs.

## Specs and keys

| spec | model | key (in .env) | notes |
|---|---|---|---|
| `openai_ocr` (default) | gpt-4o-mini | `OPENAI_API_KEY` | `image_detail:"low"` baked in — 3x fewer image tokens, right for crops |
| `gemini_ocr` | gemini-2.5-flash | `GEMINI_API_KEY` | cheapest fallback |

Cost anchor: a full 2000-frame scoreboard pass ≈ $1.30 total on openai_ocr.

## Running it

```bash
labeling-t transcribe-cloud --dataset my-ds --group all \
    --categories scoreboard --model openai_ocr --concurrency 4 --json
# read: result.transcribed, result.prefix, result.failures_file
```

- `--categories` is the REGION FILTER (which detections get cropped) — required.
- `--stems` / `--stems-file` subset for a sample-first pass.
- `--to-name ocr` writes enriched copies to `labels-ocr/` instead of rewriting
  in place (source untouched; resume reads the copy first).
- `--pad N` adds context pixels around each crop (default 2).
- Local variant: `transcribe --labels <dir> [--images <frames-dir>]` when
  labels' `image_path` doesn't resolve (e.g. from-ls output).

## Prompt override — brace escaping

`--prompt` is `.format()`ed with `categories=`, so literal `{` `}` MUST be
doubled or the command explodes:

```bash
--prompt 'Read the scoreboard. Return JSON {{"home":H,"away":A,"timer":"MM:SS"}} only.'
```

## Resume contract (why "" matters)

`Detection.text` is `None` = never attempted, `""` = attempted, nothing
legible. Re-running skips every non-None detection — so a partial/crashed run
just re-runs; already-paid crops are not re-sent. `stats` reports this as
`text.attempted` vs `text.legible`.

## Rate limits

429s are honored via Retry-After (default backoff 15/30 s). If a run crawls,
lower `--concurrency`; progress events on stderr show the real rate.

## Checkpoints

- Sample first: `--stems <5>` then `render` (text renders as the caption's
  second line) and read the values against the frames.
- Full pass: `stats --json` → `text.attempted` should approach 1.0 coverage of
  the filtered category; spot-check `failures_file`.
