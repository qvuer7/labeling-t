# Plan: keypoints labelling in Labeling-T

*Design + PR sequence. No code yet. Written 2026-07-07 against the post-agent-interface
tree (233 tests). Follows the mask playbook — the last schema extension that shipped —
rather than the speculative sibling-collection sketch in schema.py's docstring.*

## The design decision: keypoints ride on Detection

schema.py's extension plan sketched `keypoints: list[Keypoints]` as a sibling collection
on `ImageLabels`. Masks were planned the same way and DIDN'T ship like that — they ride on
`Detection.mask` because our masks are box-derived (SAM2 is box-prompted), and a
free-standing kind is only needed for promptless segmentation we don't do.

The same argument decides keypoints. The realistic first producer is **top-down pose**:
a detector finds `player` boxes, a pose model takes each box and returns its skeleton.
Keypoints are box-derived → they ride on the Detection, like `mask` and `text`:

    Detection.keypoints: list[Keypoint] | None = None     # None = never attempted
    class Keypoint:  x, y (abs px), name (str), visible (bool | None), score (float | None)

- **Named points** (`"left_knee"`), not COCO index positions — self-describing files,
  no external skeleton needed to interpret a label. Adapters map names → COCO order at
  the boundary, where such conventions belong.
- **Resume contract** mirrors text: `None` = not attempted, `[]` = attempted, nothing
  found. Per-detection resume comes free.
- **Skeleton** (names list + edge list, e.g. COCO-17) is RUN/EXPORT config — a small
  JSON spec file passed to the stage/export commands — not schema. Same reasoning as
  category_map: it's about a labeling run, not intrinsic to a label.
- Image-level keypoints (court corners for homography) are the one case this doesn't
  cover elegantly. When that dataset actually exists: either a full-frame Detection of
  category `court` carrying the points (works today under this design), or the sibling
  collection from the docstring plan. Deferred on the build-for-data-that-exists rule.

**Compat**: `extra="forbid"` means pre-change code cannot READ files containing
`keypoints` → bump `ImageLabels.schema_version` to `"2"` (only when the field is
present? No — simpler: all new dumps say "2"; loaders accept both). Old files load fine.

## Producer: `vitpose` on the existing transformers backend

`VitPoseForPoseEstimation` is transformers-NATIVE (added 4.48 < our ==4.57.1 pin — verify
with one import on the GPU box before building). Top-down, **takes boxes as input** —
the exact Sam2Model pattern: same slim GHCR image, no new deps, `MODEL=vitpose`.

- Wire: reuse the segment-style call — `params.boxes` (+ labels/scores) in,
  detections out, each carrying `keypoints: [{x, y, name, score}, ...]` in abs pixels.
  The server contract's detection dict gains an optional `keypoints` key (additive).
- ModelSpec `vitpose` (`hf_model="usyd-community/vitpose-base"`, backend transformers,
  env_prefix VITPOSE). `--concurrency 1` rule applies, as with every transformers model.

## Consumer stages (all machinery exists; this is the mask playbook re-run)

1. **`keypoints-cloud`** — enrichment stage, structural clone of `segment_cloud`:
   read label set → filter `--categories` (e.g. `player`) → box-prompt the pose pod →
   write `Detection.keypoints` in place. Gets `--to-name`, `--stems/--stems-file`,
   per-detection resume, failures sidecar, progress events for free by symmetry.
   (Neutral name: "keypoints", not "pose" — vocabulary rule.)
2. **stats** — a `keypoints` block beside `masks`/`text`:
   `{detections_with_keypoints, points_total, coverage}`.
3. **render** — draw points (small filled circles, category color) + skeleton edges
   when a `--skeleton spec.json` is given. Pixel-probe tests like boxes/masks.
4. **validate** — generalize the bounds check (docstring step 3): every keypoint
   inside width×height. diff needs NOTHING: keypoints live in the detection dump,
   so order-normalized comparison picks them up automatically.

## Label Studio round-trip — the long pole, split it

- **Import** (predictions): `KeyPointLabels` control (values in percent — geometry.py
  gains abs↔percent for points, trivial); one keypoint region per point. Generate the
  config alongside the existing rectangle/brush/polygon modes (`--keypoints` on
  `import-ls-cloud`, point labels from the skeleton spec).
- **The hard part — pull-back grouping**: LS keypoint regions are flat per-image; the
  schema needs them per-detection. Plan A: containment join (point → the box that
  contains it; ambiguity → nearest box center) — robust enough for verification passes
  where boxes are already verified. Plan B (if A proves lossy): emit `parentID` linking
  point regions to their box region at import time and read it back. Decide from a real
  10-image pilot in LS, not in the abstract.
- Until pull-back lands, keypoints are still verifiable visually in LS (accept/reject
  via `--include-accepted`), which may be all a first dataset needs.

## COCO export

`annotation.keypoints` = flat `[x,y,v,…]` in skeleton order + `num_keypoints`;
`categories[].keypoints` + `skeleton` from the spec file. supervision doesn't emit
keypoints — this is manual JSON in adapters/coco.py, and it lands together with the
already-roadmapped to-coco rework (masks/`segmentation` + cloud-aware paths). One
export overhaul, all three gaps closed.

## PR sequence (each green + committed, sized like the agent-interface PRs)

| PR | Content | Risk |
|----|---------|------|
| K1 | schema: `Keypoint`, `Detection.keypoints`, bounds validation, schema_version "2"; geometry point conversions | low |
| K2 | server: VitPose adapter (Sam2 pattern), wire contract `keypoints` key, `vitpose` ModelSpec; verify on GPU dev box first (memory: debug adapters there, not via build→rent→500) | low-med |
| K3 | `keypoints-cloud` stage + CLI (clone segment orchestration) | low |
| K4 | stats block + render points/skeleton + validate generalization | low |
| K5 | LS: import keypoint pre-annotations + config gen; pull-back grouping per pilot | **high** |
| K6 | COCO keypoints, inside the to-coco rework (with masks + cloud paths) | med |

K1–K4 give produce→inspect→verify-visually with ~zero novel design. K5/K6 are where
the unknowns live; K5 starts with a 10-image LS pilot before committing to a grouping
mechanism.

## Non-goals (now)

Bottom-up pose (OpenPose-style, needs its own grouping machinery), tracking/track_id
(separate roadmap item), video keypoints, any mmpose/ultralytics dependency in the
framework (YOLO-pose experiments stay in scripts/, per the workaround-isolation rule).
