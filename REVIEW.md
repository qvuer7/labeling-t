# Labeling-T — Architecture & Quality Review

*Reviewed: 2026-07-02, branch `transformers-model-server`, commit `a6dd9cb`, 132 tests passing.*
*Companion to [doc.md](doc.md) (what exists) and [plans/roadmap.md](plans/roadmap.md) (what's next). This file is the
honest state-of-the-union: what holds, what doesn't, and what to fix in which order.*

Standard applied: **framework code judged strictly; throwaway scripts leniently** (flagged only if
they leak into the core path — none do, see §9).

---

## 1. Verdict

The one design principle — an owned neutral schema with everything else as swappable adapters —
is **genuinely enforced, not aspirational**. The core (`schema.py`, `geometry.py`) imports no
adapter, no torch, no Label Studio; the base install is verifiably torch-free; there are three
crisp, tested seams for models, transports, and exports. Swapping Label Studio for CVAT would
touch ~4 edge files and zero core files. That is unusually clean for a CV pipeline and it is the
project's main asset.

The main debts, in order: **the schema has no version field** (the contract can't self-identify
on disk), a handful of **correctness gaps at the edges** (`.jpg` hardcoded in verified pull-back,
local-only COCO export in a cloud-first pipeline, unguarded server concurrency), and **basketball
vocabulary baked into framework defaults**, violating the project's own neutrality rule.

---

## 2. What's working (pros, with evidence)

- **Dependency inversion holds.** `schema.py` imports nothing internal; `geometry.py` imports only
  `schema`; adapters depend on core, never the reverse (grep-verified). Base deps are thin
  (pydantic/httpx/pillow/dotenv); torch lives only in the `[models]` extra installed on the pod.
  Torch-freeness is *asserted by tests*
  (`test_server_adapters.py::test_registry_returns_unloaded_adapters_without_torch`,
  `test_transformers_seam.py::test_client_path_is_torch_free`).
- **Three well-defined seams**, each with a clear contract:
  - *Transport* — `client_for(spec)` (`model_client.py:258`) picks `ChatClient` vs
    `TransformersClient`; `prelabel._raw_inference` (`prelabel.py:90`) duck-types
    `infer_raw`/`infer`, so orchestration is backend-agnostic.
  - *Registry* — `ModelSpec` frozen dataclass (`models.py:31`, `REGISTRY` at `models.py:194`).
    Adding a model = one spec entry (+ one server adapter for the transformers backend).
  - *Server* — `ModelAdapter` Protocol (`server/adapters/base.py:19`) + `StubAdapter`
    (`MODEL=stub`, no torch) backing CI. Three real adapters (OWLv2's square-pad trap,
    LocateAnything's per-category generate, SAM2's box-prompt masks) plug in without ever
    changing the wire shape — the seam demonstrably absorbs very different model shapes.
- **Swappability is real, not claimed.** LS→CVAT: rewrite `adapters/label_studio.py`, rewire
  `verify.py`, the LS CLI subcommands, and web stages — core untouched. Storage backend is a
  config choice (`open_storage`, `storage.py:153`). COCO goes through `supervision`
  (`adapters/coco.py`), never hand-written.
- **Test quality is above average.** 132 passing, no GPU/network needed. These are *contract*
  tests, not happy paths: truncated/looping model-output salvage, resume/idempotency, retry
  semantics (5xx retried, 4xx not), coordinate roundtrips (parametrized, off-canvas, zero-dim),
  strict-category → failure-manifest routing, and a real-CLI e2e (`test_cli_e2e.py`) of the
  offline spine.
- **Ops safety.** `runpod up` always sets `--terminate-after`; bare `down` refuses to guess among
  multiple pods (`AmbiguousPods`); datacenter auto-targeting replicates the web console. Creds
  ride the boto3 default chain and are never printed; `.env` is gitignored; production
  `cloud-init` generates random creds.
- **Hygiene.** Zero inline TODO/FIXME in `src/` — forward work lives in plans/roadmap.md. Scripts
  self-annotate their disposability ("throwaway, delete after PR-1"). The LS labeling-config XML
  is *generated from the category set* (`label_studio.py:55`), so config and predictions can't
  drift.

---

## 3. Data contracts (the spine)

Three contracts, each with a single home:

**Neutral schema** (`schema.py`) — the product. `_STRICT = ConfigDict(extra="forbid",
validate_assignment=True)` (`schema.py:66`) on all three models:
- `BBox` (`schema.py:69`) — abs-pixel xyxy, `Field(ge=0)`, ordering validated (`schema.py:79`).
- `Detection` (`schema.py:96`) — `bbox`, `category (min_length=1)`, `score: float|None ∈ [0,1]`
  (None = human-verified or generative), `source` provenance, `mask: dict|None` = COCO RLE
  (`schema.py:117`).
- `ImageLabels` (`schema.py:120`) — `image_path`, **required** `width`/`height`, and a validator
  that every bbox fits the image (`schema.py:136`).

**Wire contract** (`server/contract.py:20-39`) — `InferRequest`/`InferResponse`/`WireDetection`.
Pydantic-only (no torch/fastapi) so both client and server adapters import it cheaply. SAM2 box
prompts ride in `params` — no second payload type.

**Bucket layout** (`layout.py:44-63`) — `DatasetLayout` is the one naming authority:
`frames/ labels[-name]/ verified[-name]/ export/` share stems, so artifacts **join by name** and
`manifest.json` (`manifest.py`) stays a *regenerable* index, never a source of truth.

Judgement: the conventions are right (abs-px xyxy aligns with `supervision`; `extra="forbid"`
stops schema rot; dims-required makes every coord conversion safe). Coordinate math funnels
through `geometry.py`, whose percent→abs clamp (`geometry.py:96-100`) absorbs labeler edge-boxes.

**The gap: no `schema_version` field.** "v0" exists only as docstring prose. On-disk label JSON
cannot self-identify its shape, and there is no migration hook. The whole architecture bets on
this schema being the durable contract — the 946-label training set already on S3 is written in
an unversioned format. This is the single biggest architectural risk and it's a two-line fix
today; it becomes a data-archaeology problem the day the schema changes. The docstring's
horizontal-extension recipe is good discipline, but prose is not enforcement.

---

## 4. Findings — correctness (ranked; all verified in source)

1. **`verify.py:65` hardcodes `.jpg` in the pull-back join.**
   `image_path` is rewritten to `f"{frames_prefix}/{stem}.jpg"`, but ingest accepts `.png`/`.webp`
   frames. For any non-jpg dataset, verified labels silently point at nonexistent frames and the
   join-by-name breaks downstream. The frame extension should be discovered (or carried through),
   not assumed.
2. **`to-coco` is local-disk-only in a cloud-first pipeline.** The documented precondition
   (`adapters/coco.py:10-13`) is that every `image_path` exists on disk (supervision reads each
   file for dims), but cloud labels carry `s3://…` paths (`prelabel.py:329`, `verify.py:65`).
   There is no `to-coco-cloud`; exporting the cloud-verified set requires a manual sync + path
   rewrite that the framework doesn't provide. Also: the COCO adapter is export-only — no
   `from_coco` exists, so round-tripping an existing COCO dataset in is not possible.
3. **Unguarded model-server concurrency.** `/infer` is a sync `def` on one shared adapter
   instance (`server/app.py:40`), so FastAPI's threadpool will run *concurrent* `detect()` calls
   on one torch model — no lock anywhere. Meanwhile the client defaults to `max_concurrency=8`
   (`prelabel.py:232,296`). The **default configuration actively drives 8 concurrent requests
   into a non-reentrant GPU server**; safety currently depends on the operator remembering
   `--concurrency 1` (a documented gotcha in CLAUDE.md — i.e., tribal knowledge guarding a
   correctness invariant). Either the server serializes (plans/roadmap.md §2 already lists this) or the
   client should default transformers-backend runs to 1.
4. **Silent skip in LS pull-back.** `from_label_studio` drops images with zero verified regions
   because dims can't be recovered (`label_studio.py:255-257`). Intentional and documented, but
   uncounted — a labeler who legitimately deleted all boxes on a frame produces no `verified/`
   file and no signal. Should at least be counted/logged by `pull_verified`.

---

## 5. Findings — reusability / neutrality

The structure is dataset-neutral (schema, layout, `--dataset/--group/--categories` flags all
clean). The *defaults* are not:

1. **Basketball is the framework default.** Every ModelSpec ships
   `categories=("player", "ball", "referee")` (`models.py:87,107,143,160,173`), and the web UI
   placeholders repeat it (`web/static/index.html:60,72`). A neutral framework should require
   `--categories` (or default to empty and fail loudly), not silently pre-label someone's
   warehouse dataset as basketball players. This directly violates the project's own
   neutral-vocabulary rule and is the top reusability fix.
2. **Fork-hostile hardcodes.** `MODELS_IMAGE = "ghcr.io/qvuer7/labeling-t-models:latest"`
   (`runpod.py:46`) has no env override — anyone reusing the framework must edit source to point
   at their own image. Same class of issue, lower stakes: default bucket `s3://ml-cv-data`
   (`layout.py:30`; mitigated by `DatasetLayout.from_env`, but a direct construction silently
   targets Andrii's bucket), and `vllm/vllm-openai:latest` + `min_cuda="13.0"` (`runpod.py:42`,
   `gpu.py:23`) — a vLLM `:latest` bump can silently shift the CUDA floor.
3. **Softer leaks.** The only worked runbook (`runbooks/ipbl-1k-locateanything/`) is
   basketball-specific — fine as a real example, but there's no neutral quickstart beside it.
   LS-percent helpers (`abs_to_percent`/`percent_to_abs`) live in the "neutral" `geometry.py`;
   harmless, but they're adapter-flavored code in a core module. Docstring/comment examples use
   `player`/`game` in a few places (`prelabel.py:11`, `layout.py:13`, `cli.py:297`).

---

## 6. Findings — duplication & layering

- **Box clamping exists in ~4 places** with slightly different degenerate-box handling:
  `geometry.percent_to_abs`, `prelabel._box_to_bbox` (`prelabel.py:161` — note it uses
  `BBox.model_construct` to *bypass* validation pre-clamp, easy to misread), and per-adapter
  clamps in `owlv2._finalize` / `locateanything._to_wire`. The server adapters can't cleanly
  import client-side geometry across the split, so the "all coord math in geometry.py" invariant
  is partially violated by construction. Worth either accepting explicitly (document it) or
  extracting a tiny shared box-math module under `server/` reach.
- **Byte-identical retry loops** in `ChatClient.infer` (`model_client.py:129-143`) and
  `TransformersClient._post_infer` (`model_client.py:203-220`) — one shared helper.
- **Identical image-fetch blocks** in all three server adapters
  (`owlv2.py:111`, `locateanything.py:190`, `sam2.py:76`): `httpx.get(image_url, timeout=60)` with
  no retry, no max-size guard, and the server fetches **any** URL it's handed (SSRF surface —
  mitigated only by running on an ephemeral pod; still worth a shared `_fetch_image` with a
  size cap).
- **CLI-vs-web orchestration duplicated.** `verify.pull_verified` was extracted for shared use
  (its docstring says so), but prelabel-cloud and import-ls-cloud are re-implemented in
  `web/app.py` (`:146-175`, `:180-209`) vs `cli.py` (`:227-260`, `:101-121`). Next stage-shaped
  change gets made twice or drifts.
- **Layering quirk.** `RawInference` and `parse_boxes` live in orchestration-level `prelabel.py`
  but are imported *downward* by `models.py:24` and `model_client.py:32`; `CoordSpace` is defined
  in both `models.py` and `prelabel.py`. No cycle, but contract-level types belong in a small
  module below both.

---

## 7. Findings — maintenance, docs, process

- **`transformers==4.57.1` hard pin** (`pyproject.toml:45`): well-reasoned (LocateAnything's
  vendored remote code breaks on 5.x) and documented, with a dormant 5.x shim
  (`locateanything.py:66-116`) ready. But it blocks security updates to transformers on the GPU
  image, and drags **dead-weight deps** `lmdb`/`decord` (`pyproject.toml:55-56`) that exist only
  to satisfy remote-code imports never exercised at inference. Acceptable cost today; needs an
  exit strategy (drop LocateAnything, or re-vendor against 5.x) before it ages badly.
- **doc.md / plans/roadmap.md lag the code.** Both still say masks are *not* persisted in the neutral
  schema (doc.md §5/§9, plans/roadmap.md item 1) — but `Detection.mask` shipped (`schema.py:117`, commits
  `33ea14e`/`12860e0`/`a6dd9cb` closed the whole loop). Test counts stale (125 vs 132). For a
  project whose docs are this good, staleness is expensive — readers trust them.
- **The cloud loop has no e2e test.** `prelabel-cloud` / `import-ls-cloud` / `from-ls-cloud`
  wiring is `# pragma: no cover` in `cli.py`; `runpod.py` (408 lines of subprocess orchestration)
  has only small pure slices tested. The offline spine is well covered; the path actually used
  in production is the least tested. A fake-S3 (`LocalStorage`) + mocked-LS e2e would close most
  of this cheaply since the seams already exist.
- **No quality gate.** `dev` deps = pytest only; code carries ruff `# noqa`s but ruff isn't
  declared; no type-checker, no CI config in the repo. One person + discipline works until it
  doesn't; a 20-line GitHub Actions workflow (ruff + pytest) locks in the current standard.
- Minor: dev LS token baked into `docker-compose.yml:18`/`.env.example` (documented dev-only —
  fine); nginx CORS `*` is local-dev-only by design; env-var surface (`*_ENDPOINT`/`*_API_KEY`
  per model, `S3_*`, `LS_*`) is discoverable only by reading `models.py` — `.env.example` covers
  most but not all.

---

## 8. Refactor roadmap (prioritized)

**Now — cheap, high-leverage, before more data accumulates**
1. Add `schema_version` to `ImageLabels` (default `"1"`), write it on every dump, tolerate its
   absence on load (§3). Two lines now vs. migration archaeology later.
2. Fix the `.jpg` join in `verify.py:65` — carry or discover the real frame extension (§4.1).
3. Close the concurrency trap (§4.3): serialize `/infer` server-side with a lock (plans/roadmap.md §2
   already calls for this) **and/or** make `prelabel` default to concurrency 1 when
   `spec.backend == "transformers"`. Kill the tribal-knowledge dependency.
4. Neutralize category defaults in `models.py` + web placeholders (§5.1) — require categories
   explicitly; move `("player","ball","referee")` into the basketball runbook where it belongs.

**Soon — unblocks reuse and the cloud path**
5. Cloud-aware `to-coco` (§4.2): resolve `s3://` image paths through `Storage` (dims are already
   obtainable via `image_size` ranged reads) or add an export-side sync step.
6. `MODELS_IMAGE` env override (`LABELING_T_MODELS_IMAGE`) in `runpod.py:46` (§5.2).
7. Extract shared helpers: one retry loop in `model_client.py`, one `_fetch_image` (with size
   cap) for server adapters (§6).
8. Reconcile doc.md §5/§9 and plans/roadmap.md §1 with shipped mask support; fix test counts (§7).
9. Add ruff + a minimal CI workflow (§7); log skipped zero-region images in pull-back (§4.4).

**Later — structural polish**
10. Move `RawInference`/`parse_boxes`/`CoordSpace` into a low-level types module below
    `models.py`/`model_client.py` (§6).
11. Cloud-loop e2e test over `LocalStorage` + mocked LS (§7).
12. Extract shared stage orchestration so web handlers call the same functions as CLI commands
    (§6) — `pull_verified` is the template.
13. transformers-pin exit strategy (§7); already-planned items stay in plans/roadmap.md (batched `/infer`,
    cu128 image, confidence routing).

Items 3 (server serialization) and 13 overlap plans/roadmap.md; the rest are new to this review.

---

## 9. Scripts appendix (judged leniently)

No throwaway script leaks into the core path — the workaround-isolation rule is being followed.
Scripts consume `src/` as a library and self-annotate their status.

| Script | Verdict | Note |
|---|---|---|
| `spike.py` | keep | Stage-0 model smoke harness; README-referenced |
| `serve_vllm.sh` | keep | manual GPU serve recipe |
| `ls_setup.py`, `set_bucket_cors.py` | keep | real one-time ops needs |
| `curate_frames.py` | keep-ish | self-labeled "CUSTOM one-off, delete freely"; deliberately reverted out of core |
| `owlv2_debug.py`, `owlv2_smoke.py` | throwaway | self-labeled; delete when stale |
| `ls_project_local_frames.py` | throwaway | CORS workaround, obsolete once `set_bucket_cors.py` runs |
| `crop_boxes.py`, `crop_relative.py` | untracked/local | active scoreboard-OCR exploration; hardcoded absolute paths; commit or keep local, but decide |

---

*Every `file:line` in this review was verified against commit `a6dd9cb` on 2026-07-02.*
