#!/usr/bin/env bash
# Serve the student base model in the background with LoRA enabled. Every folder in adapters/ that
# has an adapter_config.json is served under the folder's name (adapters/ops_r1 -> model "ops_r1").
# Model, revision, port and parsers come from configs/models.yaml; memory settings are in
# configs/gpu.md. Extra arguments go to vllm serve and override the ones here.
set -euo pipefail
source "$(dirname "$0")/vllm_lib.sh"

lora_modules=()
for cfg in "$REPO"/adapters/*/adapter_config.json; do
  [[ -e $cfg ]] || continue
  dir=$(dirname "$cfg")
  lora_modules+=("$(basename "$dir")=$dir")
done
lora_args=()
if ((${#lora_modules[@]})); then
  lora_args=(--lora-modules "${lora_modules[@]}")
  echo "adapters: ${lora_modules[*]}"
else
  echo "adapters: none in $REPO/adapters"
fi

# Lets the factory add a newly trained adapter without a restart (POST /v1/load_lora_adapter).
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

port=$(model_field student port)
start_vllm student "$port" \
  "$(model_field student hf_id)" --revision "$(model_field student revision)" \
  --served-model-name student --port "$port" \
  --gpu-memory-utilization 0.15 --max-model-len 32768 \
  --max-num-seqs 256 \
  --enable-lora --max-lora-rank 16 --max-loras 4 "${lora_args[@]}" \
  --mamba-ssm-cache-dtype float32 \
  --enable-auto-tool-choice --tool-call-parser "$(model_field student tool_call_parser)" \
  --reasoning-parser "$(model_field student reasoning_parser)" \
  "$@"
