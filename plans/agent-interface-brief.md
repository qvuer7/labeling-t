# Brief: plan the agent-interface optimization of labeling-t

*Instruction document for a coding agent. Deliverable: a PLAN (design + prioritized
implementation steps), not code. Written 2026-07-06 after a multi-day session in which
an agent drove the full lifecycle (rent GPU → host model → pre-label → segment → OCR →
Label Studio → cleanup) through the existing CLI; every requirement below traces to real
friction from that session.*

## Goal

Make the framework operable **purely through an agent interface**: every action
invocable as a command with machine-readable input/output, no hidden mutable state,
no step that requires a human to edit files or eyeball unstructured text. The CLI
**is** the agent API — improve it; do not invent a parallel API/MCP layer unless the
plan can justify it against the constraints below.

## Hard constraints (project law — read CLAUDE.md, doc.md, REVIEW.md first)

1. The neutral schema (`schema.py`) is the contract; agent affordances must not fork it.
2. Dataset-neutral vocabulary everywhere (no domain terms in flags/commands).
3. Workaround isolation: one-off logic stays in scripts; only recurring, neutral
   patterns get promoted into the framework.
4. Thin base install (no torch); heavy deps stay in extras.
5. Human usability must not regress — agent output modes are additive.

## Known friction points to solve (evidence-based, ranked)

### 1. Endpoint state lives in `.env` mutation — the top concern
`labeling-t-runpod up` writes `{PREFIX}_ENDPOINT` into `.env`
(`runpod.py` `_write_env_endpoint`, ~line 135); clients later read it from the
environment (`models.py` `endpoint_from_env`). Problems for agents:
- `.env` is a **secrets file**; an automated process rewriting it is a bad primitive.
- Global mutable state: two pods / two sessions / parallel runs clobber each other.
- No machine-readable answer to "what is running right now, where, until when?"
Design wanted: a separate runtime-state store (e.g. `.labeling-t/pods.json` or
live-query of RunPod as source of truth) + explicit `--endpoint` override on every
inference command + `labeling-t-runpod status --json`. `.env` keeps only real secrets.
Decide and justify: state file vs live query vs both (cache w/ TTL).

### 2. No machine-readable output
Every command prints prose ("imported 1532 tasks into LS project 12 (https://…)").
Agents need `--json`: counts, output prefixes, failure-file paths, pod ids, endpoints,
LS project ids/URLs, cost/hr, terminate-at. Plan a uniform envelope (e.g.
`{"ok": true, "result": {...}}` on stdout, prose to stderr) across ALL subcommands
of both CLIs. Exit codes are already meaningful — keep them.

### 3. No dataset-state / validation primitives (agents re-derive them constantly)
This session hand-wrote the same ~30-line Python five times: validate a label set
against the schema, count detections/categories/masks/text, diff two label sets by
stem, check provenance sources. Plan first-class commands, e.g.:
- `labeling-t stats --dataset D [--labels-name N | --verified ...] --json`
  → files, detections, per-category counts, mask/text coverage, sources, schema_version mix.
- `labeling-t validate ...` → schema-validate every file, list violations.
- `labeling-t diff --a <set> --b <set>` → stems only-in-a/only-in-b/content-diff.
Also fix/extend `manifest.json` to enumerate ALL label namespaces (today it counts
only the standard prefixes and goes stale silently — see REVIEW.md).

### 4. Frame subsetting (the one seam that forced glue code twice)
`prelabel-cloud` runs over an entire group only. Recurring need: "run over the frames
appearing in label set X" (hoop-frames rim run; LS-subset staging). Plan a neutral
`--frames-from <label-set>` (and possibly `--stems <file>`) on `prelabel-cloud`
(consider `segment-cloud`/`transcribe-cloud` symmetry — they already read label sets).

### 5. Visual-verification primitive
Agents can judge rendered images (this session: caught net-in-box, purple-lighting
failure, hallucinated boxes — all visually). Each check was hand-rolled PIL code.
Plan `labeling-t render --labels <set> [--stems ...|--sample N] --out dir/` drawing
boxes/masks/text onto frames (local PNG output). Keep it in `integrations` extra (cv2/PIL).

### 6. Progress observability for long runs
Background runs emit nothing until completion (`_cmd_*` don't wire `on_progress`;
the OCR run's output file stayed empty for 40 min). Plan: periodic parseable progress
lines (`{"done": 400, "total": 2000}` to stderr or a `--progress-file`), consistent
across prelabel/segment/transcribe.

### 7. Label Studio gaps (all hit in production this week)
- `from-ls-cloud` silently exports only ANNOTATED tasks (`fetch_ls_export` omits
  `download_all_tasks`); viewed-but-unsubmitted tasks are invisible. Plan
  `--include-accepted` (falls back to the task's source prediction).
- Task-id/date-threshold slicing of verified pulls (currently a throwaway script —
  decide whether a neutral `--task-id-below` belongs in the CLI or stays scripted).
- Project title 50-char limit → validate client-side with a clear error.
- `import-ls-cloud --json` → project id + URL (agents need the link).

### 8. Cost guardrails for unattended GPU use
`--terminate-after` exists. Plan: `--max-hours`/`--budget` on `up`, surfaced in
`status --json`; refuse `up` when an equivalent pod is already running unless forced.

## Explicit non-goals
- No bespoke annotation UI, no training, no realtime (see plan.md out-of-scope).
- No speculative MCP server / Python agent-SDK layer in v1 — the plan may sketch it
  as a future thin wrapper over the `--json` CLI, but must not block on it.
- Known export-stage gaps (COCO segmentation emission, cloud-aware to-coco) are
  already tracked in plan.md §1 — reference them, don't re-plan them.

## Deliverable format
A plan document containing:
1. Design decisions for §1 (state model) and §2 (output envelope) with alternatives
   considered and rejected.
2. Per-friction-point: files/functions to change (file:line where possible), new
   flags/commands with exact signatures, test plan (this repo tests contracts, not
   happy paths — match `tests/` style, currently 157 passing).
3. A prioritized, dependency-ordered implementation sequence sized in PR-shaped
   chunks (the repo's history is small focused commits).
4. A "definition of agent-ready": a checklist an agent could follow to run the full
   lifecycle (ingest→prelabel→segment→transcribe→LS→pull→stats) with zero
   free-text parsing and zero `.env` edits — the plan's acceptance test.

Explore the codebase first (`cli.py`, `runpod.py`, `models.py`, `model_client.py`,
`prelabel.py`, `segment.py`, `transcribe.py`, `manifest.py`, `verify.py`,
`adapters/label_studio.py`); verify every claim above against source before
designing on top of it.
