# Runbook — agent-driven demo: fresh clone → Claude Code operates the whole lifecycle

The demo thesis: **the repo teaches the coding agent to operate it.** Skill,
JSON output contract, guardrails, and checkpoint discipline all ship in the
clone; only secrets travel out-of-band. The human types 4 shell commands and
then only talks to the agent. Rehearsed 2026-08-12 (source of the
extras-install and LS legacy-token fixes).

## 0. Prep (operator, before showtime)

- Secrets file ready outside any repo: `~/labeling-t.env` (S3/DO Spaces creds,
  `RUNPOD_API_KEY`, `LS_URL=https://165-245-251-248.nip.io`, `LS_API_KEY`,
  optional `OPENAI_API_KEY` for the OCR kicker).
- ~20 jpgs in `~/demo-frames` (outside the repo).
- Label Studio droplet answering: `curl -s -o /dev/null -w "%{http_code}"
  https://165-245-251-248.nip.io` → 302.
- RunPod: no stray pods, balance ≥ $5.
- Optional (kills boot dead-air, ~$0.88/hr): pre-warm both pods ~15 min before —
  `up --model locate_anything --gpu a40 --hours 3 --budget 3` and the same for
  `sam2`. The live agent then hits the duplicate-pod guardrail and reuses the
  endpoints — narrate it as the feature it is.

## 1. On screen — the only shell commands of the demo

```bash
mkdir ~/demo && cd ~/demo
git clone https://github.com/qvuer7/labeling-t.git && cd labeling-t
claude          # accept the trust prompt; skill + CLAUDE.md auto-load from the clone
```

## 2. Prompts, in order

**Self-install:**

> Set yourself up to operate this repo on this machine: install any missing
> binaries, sync dependencies, set up .env (my secrets are in ~/labeling-t.env —
> copy that file), and verify the installation is green.

Expected agent behavior (all from the skill): check/self-install `uv` +
`runpodctl`; `uv sync --extra integrations --extra cloud --extra web`;
`cp ~/labeling-t.env .env` (it never writes secret values itself);
`uv run pytest -q` → 270 passed; `labeling-t-runpod status --json`.

**Optional opener — real scale, zero risk:**

> Show me the current state of the ipbl-basketball-1k training set: stats on
> labels-combined and render 6 sample frames so we can look at the masks.

**The pipeline:**

> Ingest ~/demo-frames as dataset demo-live and get me verified
> player/ball/referee/scoreboard/hoop masks in Label Studio. locate_anything
> for boxes, sam2 for masks, a40 GPUs, $3 total budget. Sample a few frames
> first and show me the rendered result before the full run. Import with brush
> masks and give me the project URL.

Expected: ingest → (reuse or rent pod) → prelabel `--stems` sample w/
`--concurrency 1` + the runbook category phrases → render + LOOK → full run →
stats/validate → sam2 segment → stats (masks.coverage 1.0) → render →
import-ls-cloud (title ≤ 50 chars) → project URL.

**Human-in-the-loop:** open the project URL, fix one mask, submit 2–3 tasks,
just *view* a few more. Then:

> I've verified some tasks — pull everything back including accepted-as-viewed,
> show me the diff against the model's labels, then shut all pods down and
> confirm nothing is billing.

Expected: `from-ls-cloud --include-accepted --accepted-from labels` →
`diff` (changed = human fixes, byte_identical = accepted) → `down --all` →
`status` shows `pods: []`.

**OCR kicker (no GPU, ~cents):**

> OCR the scoreboards in demo-live with openai_ocr — home/away/timer into
> Detection.text — and show me a couple of results.

## 3. Fallbacks

- GPU stock dry → agent falls back per guardrails (a40 → 3090-COMMUNITY), or
  re-prompt with `gemini_vl` (hosted, boxes-only, zero GPU).
- Hosted LS misbehaves → `docker compose up -d` = local LS on `:8080`
  (legacy-token fix commented in docker-compose.yml); point `.env` at it.
- Total worst case: the opener alone still demos schema, stats, and rendered
  masks on 13k real detections.

## Talking points

- Neutral schema owns the data; models / LS / COCO are swappable adapters.
- The agent interface is deliberate: `--json` envelopes with structured
  recovery fields, `--budget`/`--hours` cost guardrails, duplicate-pod
  refusal, sample-first + render checkpoints.
- `diff` after pull-back shows exactly what the human changed — provenance
  survives the round-trip.
