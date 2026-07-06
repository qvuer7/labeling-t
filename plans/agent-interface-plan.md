# Unified plan: agent-operable Labeling-T + skill (reconciles plans A & B)

## Context

Three documents accumulated in `plans/`: the requirements brief (`agent-interface-brief.md`),
my deep plan (`agent-interface-plan.md`, "A"), and a parallel session's plan (`agent-plan.md`,
"B" — its PR-1, the `--json` envelope, is already implemented in the working tree; 166 tests
green). A and B were independently derived from the same brief and **converge on all
architectural fundamentals**: `.env` writes die; runtime pod state = local file + live
reconciliation, never a live query on the inference path; the CLI is the agent API (no MCP);
set selectors live in `layout.py`; LS accepted-tasks pull via source-copy; id-threshold rule
stays scripted. This plan merges them, resolving every divergence with recorded rationale,
and supersedes both. `plans/roadmap.md` stays the product roadmap (COCO seg export, server
serialization, etc.) — referenced, not absorbed.

## Divergence resolutions (the reconciliation record)

| # | Topic | A said | B said | Unified decision & why |
|---|-------|--------|--------|------------------------|
| 1 | State file | `.labeling-t/state.json`, dict by pod id | `.labeling-t/pods.json`, list | **`.labeling-t/pods.json`, dict keyed by pod id** — B's name (descriptive), A's shape (natural upsert/remove) |
| 2 | Reconcile | prune dead + **adopt** foreign `labeling-t-*` pods | prune only | **Adopt + prune** — pods created from console/another cwd must become usable |
| 3 | Env fallback | silent | deprecated w/ stderr note | **B**: stderr deprecation note when `{PREFIX}_ENDPOINT` is the source |
| 4 | Budget | `--budget` $ w/ delete+recreate | none (`--hours` suffices) | **`--budget` kept** (user chose hard caps) but **simplified**: pre-estimate from `runpodctl gpu list` when available; post-create check `costPerHr×hours > budget` → delete + `fail` with `suggested_hours` in the error (no recreate loop — the agent re-runs with the corrected value) |
| 5 | Test hermeticity | autouse conftest fixture for state path | not addressed | **A**: required — protects all 166 tests from a real dev state file |
| 6 | diff semantics | sha256 raw bytes | order-normalized `detections`, excluding `image_path/width/height` | **B primary, A as extra field**: `changed` compares normalized detections (verified pull-back rewrites `image_path`; byte-diff would flag everything). Result also carries `byte_identical` count — the deletion-proof rule needs bytes |
| 7 | Selector API | `layout.resolve_set(group, ref)` | `layout.set_prefix(selector, group)` | **`DatasetLayout.set_prefix(group, selector)`** — B's name, arg order matching existing `labels(group, name)` convention |
| 8 | stats module | `labelset.py` | `stats.py` | **`labelset.py`** (hosts stats+validate+diff — three commands, one domain) |
| 9 | Progress | every=25 items or 10s; --json only JSON lines | ≥5s throttle; both modes; `stage`+`elapsed_s`; `--progress-file` | **B's shape** `{"event","stage","done","total","elapsed_s"}`, throttle = first + every 25 or 5s + final, stderr in both modes, optional `--progress-file` (atomic rewrite) |
| 10 | Subsetting flags | `--stems` CSV + `--stems-file` + `--frames-from`, intersecting | `--stems <file>` XOR `--frames-from` | **A's three flags** (CSV convenience is the sample-first workflow), **B's failure semantics**: empty result set → `fail` (not silent 0-success); envelope reports `requested`/`matched` |
| 11 | Render masks | `geometry.rle_to_polygon` (needs cv2) | `pycocotools.mask.decode` → alpha composite | **B** — lighter (no cv2), simpler, decode is exactly what overlay needs; lazy import, clear fail naming `[integrations]` |
| 12 | Render edge cases | — | empty selection → fail; per-stem `failures` list, rc 0 | **B** |
| 13 | PR-0 commit series | 6 commits securing the tree | absent | **A** — mandatory opening step |
| 14 | Skill package | `.claude/skills/labeling-t/` + 3 references | absent | **A** — it's the user's stated goal |
| 15 | Docs | one docs PR at end | rolled into each PR | **B** — docs land with the code they describe |
| 16 | Acceptance | prose checklist | command sequence w/ named JSON fields per step | **B's checklist**, extended with `--budget`, render/stats checkpoints, and the skill |

## Cross-cutting architecture (agreed by both plans)

**Pod state** — new `src/labeling_t/podstate.py` (stdlib-only). File `.labeling-t/pods.json`
(gitignored, cwd-relative, atomic tmp+`os.replace`): `{"version":1,"pods":{"<id>":{id, model,
env_prefix, endpoint, gpu, cost_per_hr, created_at, terminate_after, ready}}}`.
API: `record_pod / remove_pods / load_pods / reconcile(live) / resolve_endpoint(spec, explicit=None)`.
**Resolution precedence:** `--endpoint` flag > newest unexpired pods.json entry for `spec.key`
> `{PREFIX}_ENDPOINT` env (stderr deprecation note) > `spec.default_endpoint` (SaaS specs never
touch pods.json). No runpodctl on the inference path; dead-but-unexpired pod = connection error,
recovery = `status --json` (reconciles: live list is truth for existence; prune dead ids →
`stale_removed`; adopt unknown `labeling-t-*` pods via `_proxy(id)` + name-suffix model).
`_write_env_endpoint` (runpod.py:139) and its call (:240) **deleted**; `.env` = secrets only.

**Set selectors** — `DatasetLayout.set_prefix(group, selector)` in layout.py; selector =
storage leaf name (`labels` | `labels-<name>` | `verified` | `verified-<name>`), ValueError
otherwise. Consumed by stats/validate/diff (`--set`, `--a/--b`), `--frames-from`,
`--accepted-from`, render.

**Output contract** (already shipped) — stdout: exactly one `{"ok","result"|"error"}` envelope
under `--json`; prose→stderr; ok ⇔ rc==0. Progress: JSON lines on stderr (shape per #9),
both modes.

## PR sequence

### PR-0 — Commit the tree (6 whole-file commits, unchanged from plan A)
schema → model-client → transcribe/segment stages → envelope+CLI wiring → docs → scripts.
Plus: **replace plans/agent-interface-plan.md's content with THIS unified plan and delete
plans/agent-plan.md** (superseded; note in commit message). Suite green at commit 4 and HEAD.

### PR-1 — Pod runtime state (top concern)
`podstate.py` as above; runpod.py integration (`record_pod` right after create — before the
readiness wait so a timeout still leaves state; `ready` updated on success; `stop_pods` →
`remove_pods`; `status` reconcile with `stale_removed` + per-pod `model/endpoint/terminate_after`);
`model_client.py` `from_env`×2 + `client_for` gain `endpoint: str|None` → single
`resolve_endpoint` choke point (fixes CLI + web); cli.py `--endpoint` on the 5 inference
commands; `.gitignore` += `.labeling-t/`; **tests/conftest.py** autouse `LABELING_T_PODS` → tmp.
Tests: new `test_state.py` (precedence per boundary, expiry, newest-wins, adopt/prune, corrupt
file); new `test_runpod.py` (move envelope tests from test_server_adapters.py:142-215; up writes
state + `.env` untouched; down removes; status enriches/prunes); model_client explicit-endpoint
+ state-beats-env cases. README/runbook ".env" lines updated (docs ride along).

### PR-2 — GPU guardrails
Duplicate-pod refusal in `start_pod` (deterministic name match) → new `DuplicatePod` →
`fail` with `existing={id, endpoint, cost_per_hr}` (the agent's correct move is in the payload);
`up --force` overrides. `up --budget FLOAT`: `_gpu_price()` from `runpodctl gpu list` (tolerant
None) caps hours pre-create; post-create `costPerHr×hours > budget` → delete pod + `fail`
(`cost_per_hr`, `suggested_hours` in error). Tests: refusal envelope + --force; budget caps
hours; over-budget delete+fail.

### PR-3 — Set selectors + manifest namespaces
`layout.set_prefix(group, selector)` + tests. `manifest.py`: one dataset-root scan replaces the
three `_counts_by_group` calls; legacy `groups`/`totals` byte-compatible (web/datasets.py and
test_manifest untouched); adds `namespaces: {<set>: {<group>: n}}`, `namespace_totals`,
`generated_at`. Failure files (`*_failures.jsonl`) invisible everywhere.

### PR-4 — stats / validate / diff (`src/labeling_t/labelset.py`)
Pure over the `Storage` protocol (local ≡ S3). Results:
- **stats**: `{set, prefix, files, unreadable, detections, by_category, masks:{detections_with_mask,
  files_fully_masked, coverage}, text:{attempted, legible, coverage}, sources, schema_versions}` —
  `schema_versions` counted from **raw JSON** (pydantic default masks absence); text per the
  None/"" resume contract. Empty prefix → ok:true, files:0 (query, not assertion).
- **validate**: clean → rc 0; dirty → rc 1 via `fail(..., result={files, valid,
  violations[:limit], violations_total})`.
- **diff**: join by stem; `{only_in_a, only_in_b, changed, identical, byte_identical}` —
  `changed` on order-normalized detections (sorted by (category, bbox)), excluding
  image_path/width/height; `byte_identical` = sha256 equality (deletion-proof rule).
CLI: `stats|validate [--dataset/--group/--set | --labels DIR] [--json]`;
`diff --dataset D --group G --a SEL --b SEL | --a-dir/--b-dir`. New test_labelset.py
(fixtures incl. pre-versioning + corrupt file; rewritten-path-is-not-changed diff case).

### PR-5 — Subsetting + progress wiring
`prelabel-cloud`: `--frames-from SEL`, `--stems CSV`, `--stems-file PATH` (intersection; shared
`_stem_filter(a)` in cli.py); empty intersection → `fail`; envelope gains `requested/matched`.
`segment_cloud`/`transcribe_cloud` fns gain `stems: set[str]|None` (filter after listing,
before resume check) exposed as `--stems/--stems-file`. Progress: `output.progress_reporter(a,
stage, *, every=25, min_interval=5.0)` per contract above + `--progress-file`; wired in 7
handlers (prelabel, prelabel-cloud, segment-cloud, transcribe×2, ingest-images, from-ls-cloud —
all batch fns already accept `on_progress`). Tests: subset counts via the new local-base
prelabel-cloud e2e vehicle (DatasetLayout over tmp + LocalStorage + FakeChatClient — first e2e
coverage for prelabel-cloud); throttle unit tests (mock monotonic); stderr progress + stdout
single-envelope invariant; progress-file holds final event.

### PR-6 — render (`src/labeling_t/render.py`)
`render_labels(labels, image_bytes) -> bytes` (pure PNG-in/PNG-out; boxes + `category score`
caption + `Detection.text` second line, stable per-category colors) and `render_set(prefix, *,
storage, out_dir, stems=None, sample=None, seed=0, on_progress=None) -> dict`. Masks:
`pycocotools.mask.decode` → 25%-alpha composite, lazy import, missing dep → `fail` naming
`[integrations]`; boxes/text = base install (PIL). Deterministic sampling; output always local
PNGs; envelope `{rendered, out, stems, failures:[{stem,error}]}` — bad RLE/missing frame →
failures entry rc 0; empty selection → `fail`. CLI local + cloud modes per plan A. Pixel-probe
tests, no golden files; `--sample 2 --seed 0` twice → identical stems.

### PR-7 — LS `--include-accepted`
`verify.fetch_ls_export(..., all_tasks=False)` (`download_all_tasks=true`, timeout 300);
`pull_verified(..., include_accepted=False, accepted_from="")` (required together); accepted
tasks = `storage.copy` from `set_prefix(group, accepted_from)` (export carries prediction IDs,
not bodies — the source file IS the truth; proven on project 11); missing sources collected,
not fatal. Return `int` → `{"pulled","corrected","accepted","missing_source"}`; update cli.py +
web/app.py callers + test_web. New test_verify.py (monkeypatched export + LocalStorage;
MockTransport for param assembly). Script header marked superseded-except-id-threshold.

### PR-8 — Skill + final docs sweep
`.claude/skills/labeling-t/SKILL.md` (~150 lines, thin) + `references/{lifecycle,
ocr-enrichment, guardrails}.md` per plan A: prerequisites check (`status --json` at session
start; secrets-only `.env`), envelope contract ("never parse prose"), command map, **checkpoint
discipline** (stats + render + look after every stage; sample-first-then-batch via `--stems`;
diff before believing a rewrite; never delete without `byte_identical` proof), guardrails
(`--budget`/`--hours` always; duplicate-refusal → reuse `error.existing.endpoint`;
`--concurrency 1` transformers; LS ≤50 chars; presigned 7-day TTL; OCR brace-doubling).
lifecycle.md = plan B's acceptance checklist verbatim as the worked recipe. doc.md gains the
"Agent interface" section; CLAUDE.md gains the agent runbook + drops stale gotchas; REVIEW.md
manifest finding marked resolved. plans/ ends with: brief, unified plan, roadmap.md.

## Dependencies
PR-0 → PR-1 → PR-2; PR-3 independent after 0; PR-4/5 need 3; PR-6 needs 4 (selector) + 5
(stem filter); PR-7 needs 3; PR-8 last. Suite green after every PR (166 now → ~230 expected).

## Acceptance (definition of agent-ready — plan B's checklist, extended)
The exact sequence an agent must complete with zero free-text parsing, zero `.env` edits:
`datacenters --json` → `up --model locate_anything --gpu a40 --budget 3 --json`
(reads result.id/endpoint/ready/terminate_after; on rc≠0 reads error.existing.id) →
`ingest-images --json` → `prelabel-cloud --stems <5> --json` → `render --sample 8` + LOOK →
full `prelabel-cloud` (stderr progress events) → `stats`/`validate --json` (rc 0) →
pod swap exercising duplicate-refusal → `segment-cloud --json` → `render` →
`import-ls-cloud --json` (result.project_url) → `from-ls-cloud --include-accepted
--accepted-from <sel> --json` → `diff --a <sel> --b verified --json` → `down --json` +
`status --json` confirms pods==[]. Run live on a 10-image dataset as the final check.

## Verification
- `uv run pytest -q` green after every PR.
- The acceptance sequence above executed end-to-end by an agent (this one) on a small dataset.
- Grep-proof after PR-1: no `_write_env_endpoint`, no `.env` writes outside human editing.
- After PR-8: fresh Claude Code session invokes the skill and completes the lifecycle unaided.
