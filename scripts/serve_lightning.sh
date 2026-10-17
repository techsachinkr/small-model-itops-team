#!/usr/bin/env bash
# Serve Nemotron 3.5 Lightning in the background with the 1x H100 flags from its model card.
# Model, revision, port and parsers come from configs/models.yaml; memory settings are in
# configs/gpu.md. Extra arguments go to vllm serve and override the ones here.
set -euo pipefail
source "$(dirname "$0")/vllm_lib.sh"

port=$(model_field lightning port)
start_vllm lightning "$port" \
  "$(model_field lightning hf_id)" --revision "$(model_field lightning revision)" \
  --served-model-name lightning --port "$port" \
  --gpu-memory-utilization 0.55 --max-model-len 32768 \
  --max-num-seqs 128 --enable-prefix-caching --async-scheduling \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 \
  --enable-auto-tool-choice --tool-call-parser "$(model_field lightning tool_call_parser)" \
  --reasoning-parser "$(model_field lightning reasoning_parser)" \
  --structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}' \
  "$@"
