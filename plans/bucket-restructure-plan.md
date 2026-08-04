# ml-cv-data bucket restructure — plan v2 (2026-08-04)

**Status: AGREED IN PRINCIPLE — reviewed by the ipbl training agent; their
three corrections + question answers are folded in below. Nothing moved yet.**
Execution is gated per the sequencing section; every move is
**copy → verify → repoint → delete**, never a bare `mv` (the 07-20 seg move
left a dead fetch-script path for a week — that failure mode is designed out).

## Goal

Efficient storage of everything that is **labelled or partially labelled**:
for any data point that has been labelled *in any way* (box, mask, keypoints,
OCR text, verified or model-only), that label must be **retrievable by
addressing, not by archaeology** — one query, no reading CLAUDE.md prose, no
guessing prefixes.

Two mechanisms deliver that:
1. **Zones + one strict contract** — labels only ever live under
   `datasets/<name>/` in `layout.py` shape; same filename stem = same data
   point across frames/labels/verified sets.
2. **A queryable index** — manifest v2 covering *all* label sets (today the
   manifest ignores `labels-<name>`/`verified-<name>`, where most real labels
   live), plus bucket-level listing and per-stem lookup commands.

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
                                 trained on. NEVER holds sealed-eval data.
  models/<model>/<run>/          TRAINED MODELS — single home (see weights note)
  experiments/<project>/         WORKSPACES (offroad-seg pile)
  eval/<task>/<run>/             MIXED — see eval retention classes below
  demos/                         rendered deliverables
  files-exchange/                scratch/handoff — purge-eligible wholesale
                                 (ipbl agent confirms nothing load-bearing)
  bucket-manifest.json           zone declaration, discoverable by any agent
```

Rule of thumb: **`datasets/` is contract, everything else is convention.**

### eval/ retention classes (Correction 2 — NOT uniformly deletable)

- **KEEP FOREVER — sealed-run evidence**: `newdata_run`, `sealed3_run`,
  `sealed4_run` (+ future sealed runs). The quarantine log's rule: re-evaluation
  reads these artifacts, not the videos. They are the evidence behind one-shot
  numbers. Treat as immutable.
- **KEEP — frozen baselines**: `pro_run` (July baseline record),
  `2026-07-20-baseline`.
- **REZONE — human labels misfiled here**: `scored-missed*/miss_labeling.csv`,
  `error_corpus/`, `vision_label_new/` → these are labels and belong under
  `datasets/<d>/labels-*` by this plan's own contract. Move them INTO the
  labeling zone; do not let any cleanup rule eat them.
- **Deletable once conclusions recorded**: everything else (ordinary run
  outputs).

### Sealed data & quarantine (Correction 1)

`test_rich28` (crops of sealed 1079351/355/362) and `sealed3_rich28` (sealed3
grading archives) are **sealed-eval evidence, not training data** — they do
NOT go under `training/`. They move under `eval/ipbl-rim-crops/` (keep-forever
class). Additionally:

- **Manifest v2 gains a mandatory `tier` field** per set:
  `train | validation | sealed | donor`, sourced from the ipbl repo's
  `docs/EVAL_QUARANTINE.md`.
- The ipbl repo's `reject_sealed` guard reads the manifest as a **second
  source of truth**, so quarantine is enforced by addressing, not vigilance.

## Retrieval story

- **Join rule (exists):** `frames/<g>/<stem>.jpg` ↔ `<set>/<g>/<stem>.json`.
- **Stem convention (Q4 — formalize):** `<game>_f<frame>` is already the
  de-facto stem in court-gt, review clips, and the error corpus. Formalize it
  with the contract caveat: **frame indices are only meaningful against the
  canonical whole-game concat** (fetch_streams.sh order, sequential decode);
  the VFR trap makes any other indexing wrong. Document in layout.py.
- **Manifest v2 (build):** per-set entries for ALL sets:
  `{set, group, count, categories, detections, masked%, has_text,
  has_keypoints, tier, note}` — `note` moves meaning ("THE training set",
  "deletable leftovers") out of CLAUDE.md prose; `tier` carries quarantine.
- **`labeling-t datasets` (build):** bucket-level listing, one command answers
  "what's labelled on this bucket".
- **`labeling-t locate --dataset <d> --stem <s>` (build):** every set holding
  a label for that stem, with per-set summary (boxes n, mask y/n, text y/n,
  keypoints y/n).

## Weights/models consolidation (Correction 3 — direction REVERSED)

Live-reference counts: `models/*` → 9 files (fetch scripts, both agents' push
paths); `weights/ipbl-basketball-seg` → 2 files. The 07-20 "move to weights/"
was a side effect of ultralytics run dirs, not a decision. So consolidate the
cheap way: **`weights/ipbl-basketball-seg/` → `models/ipbl-basketball-seg/`**
(2-file patch), delete `weights/`, `models/` is the single home.
(If Andrii prefers the `weights/` name anyway: ipbl agent has offered to patch
`fetch_ipbl_models.sh`/`fetch_ipbl_assets.sh` in lockstep — but default is
models/.) Note: `models/` has 7 dirs incl. `ipbl-reid/` (v0_A..E, v1_E,
v1_E_long) — inventory below.

## Migration map (updated per Q1/Q3 answers)

Reference counts are from the ipbl repo (code only); a committed old→new path
map accompanies every batch so both repos migrate in lockstep.

| # | move | refs to patch | gate |
|---|---|---|---|
| 1 | `weights/ipbl-basketball-seg/` → `models/ipbl-basketball-seg/` | 2 files | anytime |
| 2a | `datasets/ipbl-court-pose/images/` → `frames/` | part of 8 court-pose refs | anytime |
| 2b | `datasets/ipbl-rim-crops/label-v1/` → `labels-v1/`, `label-v2/` → `labels-v2/` | 1 file (ff5 --labels-csv default) | anytime |
| 2c | `datasets/ipbl-player-pose/pseudo-v1/` → `labels-pseudo-v1/`, `verify-v1/` → `verified-v1/` | 4 files | anytime |
| 3 | top-level `offroad-seg/` → `experiments/offroad-seg/` | 0 refs | anytime |
| 4 | `datasets/ipbl-court-gt/` → `training/ipbl-court-gt/` | 8 files (court-GT factory + vlad bench) | sprint pause |
| 5 | `datasets/ipbl-reid/` → `training/ipbl-reid/` | active v2 work | **sprint pause (hot)** |
| 6 | rim-crops split: `rich28`, `newdata_rich28`, `v7_rich28`, `v7_newdata_rich28`, `pretrain_frames` → `training/ipbl-rim-crops/…`; `test_rich28`, `sealed3_rich28` → `eval/ipbl-rim-crops/…` (sealed evidence); `labels-*`, `ball_sidecars` stay in datasets/ | 8 files (ff5/6/7 trainers, slide/ablate) | **sprint pause (ff7 writes daily)** |
| 7 | court-pose 3-way split: YOLO export (`dataset.yaml` + yolo dirs) → `training/ipbl-court-pose/`; `seed-arc-v1/` stays (labeling zone, as `labels-seed-arc-v1/`); `vlad-far-bench/` → `eval/court-pose/vlad-far-bench/` (frozen bench) | within the 8 court-pose refs | sprint pause |
| 8 | rezone misfiled human labels out of eval/ (scored-missed csv, error_corpus, vision_label_new) → `datasets/<d>/labels-*` | check both repos | anytime, carefully |
| 9 | purge `files-exchange/` (confirmed nothing load-bearing) | 0 | after user nod |

Known dedup decision queued (NOT this plan): `v7_rich28`/`v7_newdata_rich28`
(~1 GiB) near-duplicate rich28 — revisit post-ff7.

## Sequencing

- **Anytime:** #1–3, #8, index code (manifest v2, `datasets`, `locate`,
  bucket-manifest.json). Rename batches land same-window with the ipbl agent's
  1–2-file patches.
- **Sprint pause only:** #4–7 — anything touching `rich28*`,
  `models/ipbl-rim-crop` (ff7 writes daily) or `ipbl-reid` (active v2).
- **Protocol for every move:** copy → verify (count + spot-check bytes) →
  repoint (both repos, committed path map) → delete old prefix. Never bare mv.
- **After migration:** rebuild all manifests, write `bucket-manifest.json`,
  trim CLAUDE.md Cloud-state prose to decisions/history only.

## Resolved review points (log)

- ✅ Zones + contract + index: agreed by ipbl agent.
- ✅ C1 sealed quarantine: sealed sets → eval/ + mandatory manifest `tier`
  field + `reject_sealed` reads manifest.
- ✅ C2 eval retention classes: encoded above.
- ✅ C3 weights direction: reversed — models/ is home (2-file vs 9-file patch).
- ✅ Q1 inventory corrections: v7_rich28 + v7_newdata_rich28 exist (active ff7
  inputs); models/ has 7th dir ipbl-reid. Fresh full listing available from
  ipbl agent on request.
- ✅ Q2: files-exchange purge-eligible; eval per retention classes.
- ✅ Q3: court-pose split 3 ways (training/labeling/eval).
- ✅ Q4: formalize `<game>_f<frame>` stem + whole-game-concat caveat.
