# ml-cv-data bucket restructure — plan (2026-08-04)

**Status: PROPOSAL — nothing moved yet.** Location/naming changes below need
sign-off from the agent working on ipbl training/experiments before any
`aws s3 mv` runs, because moves break hardcoded paths in that work.

## Goal (the honest one)

Efficient storage of everything that is **labelled or partially labelled**:
for any data point that has been labelled *in any way* (box, mask, keypoints,
OCR text, verified or model-only), that label must be **retrievable by
addressing, not by archaeology** — one query, no reading CLAUDE.md prose, no
guessing prefixes.

Two mechanisms deliver that:
1. **Zones + one strict contract** — labels only ever live under
   `datasets/<name>/` in `layout.py` shape, so there is exactly one place to
   look and one join rule (same filename stem = same data point across
   frames/labels/verified sets).
2. **A queryable index** — the manifest, extended to cover *all* label sets
   (today it ignores `labels-<name>`/`verified-<name>`, which is where most
   real labels live), plus a bucket-level listing command.

## Current state (live listing, 2026-08-04)

Conforming to `layout.py` (labels retrievable today):

| prefix | sets | note |
|---|---|---|
| `datasets/ipbl-basketball-1k` | labels-combined (1532, THE training set), labels-ocr-clean (2000), labels-hoop, labels-yolo-sam2, verified | fully conformant |
| `datasets/ipbl-basketball-seg` | labels (flat, group ""), labels-v2, labels-sam3-court-final (+ deletable prompt-iteration leftovers) | conformant |
| `datasets/ipbl-scoreboard-kp`, `ipbl-court-kp` | seeded keypoint sets | conformant |
| `datasets/offroad-seg` | frames/ + labels-v15 | conformant except stray `classes.json` at root |
| `streams/<match-id>/` | — | raw zone, immutable, fine as-is |

NOT conforming (labels exist but are not framework-addressable):

| prefix | what it is | problem |
|---|---|---|
| `datasets/ipbl-court-pose` | images/ + labels/ + dataset.yaml + seed-arc-v1/ + vlad-far-bench/ | `images/` not `frames/`; YOLO export mixed into labeling dataset |
| `datasets/ipbl-player-pose` | pseudo-v1/, verify-v1/ | home-grown stage names instead of `labels-*` / `verified-*` |
| `datasets/ipbl-rim-crops` | label-v1/, label-v2/, ball_sidecars/, rich28/, sealed3_rich28/, test_rich28/, newdata_rich28/, pretrain_frames/ | 7 ad-hoc prefixes; `label-v1` ≠ `labels-v1` so set selectors reject it |
| `datasets/ipbl-court-gt` | classic/, human_v3/, v3/, *_holdout/, selfharvest/ | versioned *training* dataset, not a label-set layout |
| `datasets/ipbl-reid` | v1/, v2/, group_*/ | same — training dataset |
| `models/` vs `weights/` | 6 model dirs vs 1 | split-brain; move to weights/ was decided 2026-07-20 but models/ kept growing |
| `offroad-seg/` (top level) | checkpoints, embeddings, experiments_* | whole experiment workspace shadowing `datasets/offroad-seg` |
| `files-exchange/`, `eval/`, `inference/`, `demos/` | scratch, run outputs, kits.json, renders | no rules declared (mostly fine, just undocumented) |

## Target structure — zones

```
s3://ml-cv-data/
  streams/<match-id>/            RAW — immutable source of truth (unchanged)
  datasets/<name>/               LABELING — STRICT layout.py contract:
      frames/<group>/<stem>.jpg          only these prefixes allowed:
      labels[-<set>]/<group>/<stem>.json frames | labels[-x] | verified[-x]
      verified[-<set>]/<group>/          | export/ | manifest.json
      export/<version>/                  anything else = lint error
      manifest.json
  training/<name>/<version>/     TRAINING SETS — versioned, immutable once
                                 trained on (court-gt, reid, rich28 family)
  weights/<model>/<run>/         TRAINED MODELS — single home
  experiments/<project>/         WORKSPACES (offroad-seg pile)
  eval/<task>/<run>/             RUN OUTPUT — deletable once conclusions recorded
  demos/                         rendered deliverables
  files-exchange/                scratch / human handoff — deletable anytime
  bucket-manifest.json           zone declaration, discoverable by any agent
```

Rule of thumb: **`datasets/` is contract, everything else is convention.**
A label that isn't under `datasets/<d>/labels*|verified*` does not exist as
far as retrieval is concerned — that's the discipline that makes the goal hold.

## Retrieval story (what "easy to retrieve" concretely means)

- **Join rule (exists today):** frame `frames/<g>/<stem>.jpg` ↔ label
  `<set>/<g>/<stem>.json`. Same stem = same data point in every set.
- **Manifest v2 (to build):** per-dataset `manifest.json` gains per-set
  entries for ALL sets incl. named ones: `{set, group, count, categories,
  detections, masked%, has_text, has_keypoints, note}` — the `note` field
  moves meaning ("THE training set", "deletable leftovers") out of CLAUDE.md
  prose into the index.
- **`labeling-t datasets` (to build):** bucket-level listing of every dataset
  + its manifest summary. One command answers "what's labelled on this bucket".
- **`labeling-t locate --dataset <d> --stem <s>` (to build, small):** list
  every set containing a label for that stem, with per-set summary (boxes n,
  mask y/n, text y/n, keypoints y/n). This is the literal "give me everything
  we know about this frame" query.
- Cross-dataset identity (1k vs seg share physical frames) — see open
  questions; not solved by this plan, only made visible by it.

## Migration steps (cheapest → heaviest; NONE executed yet)

Every step: grep this repo AND the ipbl training repo(s) for the old prefix
before moving; `aws s3 mv --recursive` after sign-off; update manifest after.

1. **weights split-brain**: `models/<m>/` → `weights/<m>/` (6 dirs:
   court-heatmap, player-pose, reid, rim-crop, scoreboard-ocr, shot-vision).
2. **rename-only conformance** (framework gains addressing, zero semantic change):
   - `datasets/ipbl-court-pose/images/` → `frames/`
   - `datasets/ipbl-rim-crops/label-v1/` → `labels-v1/`, `label-v2/` → `labels-v2/`
   - `datasets/ipbl-player-pose/pseudo-v1/` → `labels-pseudo-v1/`,
     `verify-v1/` → `verified-v1/`
3. **rezone training-shaped data**:
   - `datasets/ipbl-court-gt/` → `training/ipbl-court-gt/`
   - `datasets/ipbl-reid/` → `training/ipbl-reid/`
   - `datasets/ipbl-rim-crops/{rich28,sealed3_rich28,test_rich28,newdata_rich28,pretrain_frames}/`
     → `training/ipbl-rim-crops/…` (the labels-* sets STAY in datasets/)
   - `datasets/ipbl-court-pose/{dataset.yaml,seed-arc-v1,vlad-far-bench}` →
     decide: training/ or eval/ (ask ipbl agent)
4. **rezone workspace**: top-level `offroad-seg/` → `experiments/offroad-seg/`.
5. **index work (code, this repo)**: manifest v2 → `labeling-t datasets` →
   `locate` → write `bucket-manifest.json`. Then trim CLAUDE.md's Cloud-state
   prose to decisions/history only.

## Open questions for the ipbl agent

1. Which prefixes under `ipbl-court-gt`, `ipbl-reid`, `ipbl-rim-crops`,
   `ipbl-court-pose` are referenced by live training configs/scripts, and
   are the proposed names acceptable?
2. Is anything in `files-exchange/` or `eval/` still load-bearing, or is it
   all reproducible/concluded?
3. `datasets/ipbl-court-pose/labels/` — neutral-schema JSON or YOLO txt? If
   YOLO, it belongs in training/, not datasets/.
4. Cross-dataset frame identity: do we want a canonical frame ID
   (`<match>_<segment>_<frame>`?) so labels for the same physical frame in
   ipbl-basketball-1k and ipbl-basketball-seg can be joined? (Out of scope
   here, but the `locate` command would surface the duplication.)
