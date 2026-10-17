# GPU plan

One NVIDIA H200 (141 GB) on Brev, shared by every server and by training. Numbers marked "est." are estimates to replace with measurements from task 0.2 (`nvidia-smi` with each server loaded).

| Process | Port | Memory (est.) | vLLM `--gpu-memory-utilization` | Notes |
|---|---|---|---|---|
| Lightning (BF16), vLLM | 8000 | about 75–80 GB | _set in 0.2_ | about 60 GB of weights; the rest is KV and Mamba cache |
| Student + LoRA adapters, vLLM | 8001 | about 15–20 GB | _set in 0.2_ | `--enable-lora`; max LoRA rank 16 |
| LoRA training (TRL + PEFT) | – | about 20–30 GB | – | runs next to both servers if memory allows |

## Rules

- vLLM reserves memory by fraction of the whole GPU, so the two servers' fractions must add up to less than 1 with room left for training.
- If training doesn't fit next to both servers, stop the student server (`scripts/stop_vllm.sh`) while training, and keep Lightning up.
- Record the final flags and measured memory here once task 0.2 has them.
