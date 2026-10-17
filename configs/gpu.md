# GPU plan

One NVIDIA H200 (143,771 MiB as reported by `nvidia-smi`) on Brev, shared by every server and by training. Measured in task 0.2 (2026-10-17) with both servers loaded and idle; the training row is still an estimate.

| Process | Port | Measured memory | `--gpu-memory-utilization` | Notes |
|---|---|---|---|---|
| Lightning (BF16), vLLM | 8000 | 78,230 MiB (78,994 on the verify run) | 0.55 | weights 58.92 GiB, KV cache 14.98 GiB (513,440 tokens), CUDA graphs 0.43 GiB |
| Student + LoRA adapters, vLLM | 8001 | 21,644 MiB | 0.15 | weights 7.71 GiB, KV cache 11.36 GiB (624,230 tokens), CUDA graphs 0.71 GiB |
| Both servers | | 99,889 MiB | 0.70 | about 43,900 MiB left on the GPU |
| LoRA training (TRL + PEFT) | – | about 20–30 GB (est.) | – | should fit in the 43.9 GB left; measure in task 2.2 |

## Server settings (`scripts/serve_*.sh`)

| | Lightning | Student |
|---|---|---|
| `--max-model-len` | 32768 (card validates 256K) | 32768 |
| `--max-num-seqs` | 128 (card) | 256 |
| Prefix caching | on (card) | vLLM default for this model: off |
| Mamba SSM cache | float16, stochastic rounding, 5 Philox rounds (card) | float32 (card) |
| Mamba backend | flashinfer (card) | vLLM default (triton) |
| LoRA | – | `--enable-lora --max-lora-rank 16 --max-loras 4`, runtime loading on |
| Structured outputs | xgrammar, compact JSON (`disable_any_whitespace`) | same |
| Startup time | 135 s | 40 s (the first start compiles kernels and takes longer) |

Compact JSON (task 1.2): with any whitespace allowed, a Lightning self-check answer got stuck after `{"complete": true` and generated whitespace until `max_tokens` (about 8k tokens). `disable_any_whitespace` needs the backend named explicitly; `auto` is rejected.

At 0.15 the student has 643 Mamba cache blocks, and vLLM needs one per running sequence. Its default `--max-num-seqs` of 1024 doesn't fit and stops startup, hence 256. Raise the share before raising `--max-num-seqs` past 643.

## Rules

- `--gpu-memory-utilization` is each instance's share of the whole GPU, so the two servers' shares must add up to less than 1, with room left for training.
- If training doesn't fit next to both servers, stop the student server (`scripts/stop_vllm.sh student`) while training, and keep Lightning up.
- Extra arguments to a serve script go to `vllm serve` and override its flags (for example a different `--gpu-memory-utilization` for a benchmark in task 4.1).
