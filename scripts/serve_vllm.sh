#!/usr/bin/env bash
# Run this ON THE RENTED GPU BOX (CUDA + ~16GB+ VRAM for a 3B VLM).
# It serves the model behind an OpenAI-compatible API on :8000.
set -euo pipefail

pip install -U vllm

# --- Option A: LocateAnything-3B (the chosen v0 model) ------------------------
# trust-remote-code because it's a custom NVIDIA arch. If vLLM rejects the
# architecture, it isn't supported yet on your vLLM version — use Option B to
# validate the pipeline, then revisit.
vllm serve nvidia/LocateAnything-3B \
  --served-model-name locate-anything-3b \
  --trust-remote-code \
  --port 8000 \
  --limit-mm-per-prompt image=1
  # --api-key YOUR_SECRET     # optional; if set, put the same value in .env

# --- Option B: Qwen2.5-VL (fallback, definitely vLLM-supported) ---------------
# Use this ONLY if Option A fails, to prove the end-to-end loop works; then swap.
# vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
#   --served-model-name qwen2.5-vl \
#   --port 8000 \
#   --limit-mm-per-prompt image=1

# --- Reaching it from the laptop ---------------------------------------------
# If the GPU box has a public IP: set LABELING_T_ENDPOINT=http://PUBLIC_IP:8000
# If not, tunnel from the laptop:  ssh -N -L 8000:localhost:8000 user@gpu-box
#   then LABELING_T_ENDPOINT=http://localhost:8000
