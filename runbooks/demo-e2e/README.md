# Runbook — fresh-clone demo: 20 frames → boxes → masks → Label Studio → verified

The live demo script: clone from GitHub on a "fresh" machine, install, run the
full lifecycle on ~20 basketball frames with a hosted Label Studio. Rehearsed
2026-08-12 (which is where the extras-install and LS-token fixes came from).

## 0. Prep (before showtime)

- **Label Studio droplet up** (`165.245.251.248`): power on in the DO console,
  then `curl -s -o /dev/null -w "%{http_code}" https://165-245-251-248.nip.io`
  → 200/302. Fallback: local `docker compose up -d` (see the legacy-token
  gotcha commented in docker-compose.yml).
- **RunPod**: no stray pods (`labeling-t-runpod status --json`), balance ≥ $5.
- **Frames**: ~20 jpgs in a folder OUTSIDE the repo (e.g. `~/demo-frames`).
- **Pre-warm both pods ~15 min before** (kills all boot dead-air; ~$0.88/hr):

```bash
uv run labeling-t-runpod up --model locate_anything --gpu a40 --hours 3 --budget 3 --json
uv run labeling-t-runpod up --model sam2            --gpu a40 --hours 3 --budget 3 --json
```

## 1. Fresh install (live, ~4 min)

```bash
git clone https://github.com/qvuer7/labeling-t.git && cd labeling-t
cp <your-saved>/.env .env       # secrets travel out-of-band; see table below
uv sync --extra integrations --extra cloud --extra web
uv run pytest -q                # → 270 passed
claude                          # skill + CLAUDE.md auto-load from the repo
```

`.env` keys required for this demo (values from the operator's saved copy):

| key | value |
|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | DO Spaces creds |
| `S3_ENDPOINT_URL` / `S3_REGION` / `S3_BUCKET` | `https://fra1.digitaloceanspaces.com` / `fra1` / `ml-cv-data` |
| `RUNPOD_API_KEY` | RunPod account key |
| `LS_URL` | `https://165-245-251-248.nip.io` |
| `LS_API_KEY` | the droplet LS token |
| `OPENAI_API_KEY` (optional) | OCR kicker only |

## 2. The lifecycle (either tell Claude the goal and let the skill drive, or by hand)

Agent one-liner: *"Ingest ~/demo-frames as dataset demo-live and get me verified
player/ball/referee/scoreboard/hoop masks in Label Studio — sample first."*

By hand:

```bash
uv run labeling-t-runpod status --json          # pods ready + balance
uv run labeling-t ingest-images --src ~/demo-frames --dataset demo-live --group all --json

# sample 5 stems first — checkpoint discipline
uv run labeling-t prelabel-cloud --dataset demo-live --group all --model locate_anything \
  --categories "basketball player,basketball,referee,scoreboard,basketball hoop" \
  --category-map runbooks/ipbl-1k-locateanything/category_map.json \
  --concurrency 1 --stems <s1,s2,s3,s4,s5> --json
uv run labeling-t render --dataset demo-live --group all --set labels --out /tmp/r1 --json
# LOOK at the PNGs, then the full run: same command minus --stems (~8 s/frame)

uv run labeling-t stats    --dataset demo-live --group all --set labels --json
uv run labeling-t validate --dataset demo-live --group all --set labels --json

uv run labeling-t segment-cloud --dataset demo-live --group all --concurrency 1 --json
uv run labeling-t stats  --dataset demo-live --group all --set labels --json  # masks.coverage 1.0
uv run labeling-t render --dataset demo-live --group all --set labels --sample 8 --out /tmp/r2 --json
uv run labeling-t-runpod down --all --json      # stop billing the moment stages finish

uv run labeling-t import-ls-cloud --dataset demo-live --group all --project "demo-live masks" \
  --categories "player,ball,referee,scoreboard,hoop" --masks --mask-format brush --json
# open result.project_url → fix a mask, submit a few, just VIEW the rest

uv run labeling-t from-ls-cloud --dataset demo-live --group all --project-id <ID> \
  --name demo --include-accepted --accepted-from labels --json
uv run labeling-t diff --dataset demo-live --group all --a labels --b verified-demo --json
# result.changed = human fixes · result.byte_identical = accepted as-is

uv run labeling-t-runpod status --json          # confirm pods: []
```

Optional OCR kicker (no GPU, ~cents):

```bash
uv run labeling-t transcribe-cloud --dataset demo-live --group all \
  --categories scoreboard --model openai_ocr \
  --prompt 'Return ONLY JSON: {{"home": <int>, "away": <int>, "timer": "<clock>"}}' --json
```

## Talking points woven through

- Neutral schema owns the data; models / LS / COCO are swappable adapters.
- Guardrails: `--budget` hard cap, `--hours` backstop, duplicate-pod refusal
  (run `up` twice to show it), `--json` envelope contract, sample-first + render.
- `diff` after pull-back shows exactly what the human changed — provenance intact.
