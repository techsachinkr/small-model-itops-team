# GPU plan

One NVIDIA H200 (143,771 MiB as reported by `nvidia-smi`) on Brev, shared by every server and by training. Server rows measured in task 0.2 (2026-10-17) with both servers loaded and idle; the training row in task 2.2 (same day), next to both servers.

| Process | Port | Measured memory | `--gpu-memory-utilization` | Notes |
|---|---|---|---|---|
| Lightning (BF16), vLLM | 8000 | 78,230 MiB (78,994 on the verify run) | 0.55 | weights 58.92 GiB, KV cache 14.98 GiB (513,440 tokens), CUDA graphs 0.43 GiB |
| Student + LoRA adapters, vLLM | 8001 | 21,644 MiB | 0.15 | weights 7.71 GiB, KV cache 11.36 GiB (624,230 tokens), CUDA graphs 0.71 GiB |
| Both servers | | 99,889 MiB | 0.70 | about 43,900 MiB left on the GPU |
| LoRA training (TRL + PEFT) | – | up to about 30,000 MiB | – | `factory/train_lora.py`, batch 2 x 8: full runs peaked at 27,582 MiB reserved by PyTorch (helpdesk) and 26,718 (ops), 16.5 GB allocated; the dry runs' 24,306 was one step, before the allocator's cache grew. `--api all` (task 3.2; every tool in the prompt, up to 4,790 tokens): 30,238 reserved, 18.9 GB allocated |

Task 2.2: training runs next to both servers. In task 2.2 they used 105,647 MiB (Lightning 81,104, student 24,528), more than in task 0.2, leaving 38,124. With training at its worst-case batch, the GPU peaked at 130,724 MiB, leaving 13 GB, in the dry run; the full runs reserved up to 27,582 MiB of the 36,984 free when they started, so about 9 GB stayed free. Task 3.2's `--api all` run, with longer prompts, reserved 30,238 MiB of the 37,644 free at its start (about 7 GB left); its one-step dry run peaked the GPU at 130,164 MiB. Batch 4 x 4 peaked at 142,418 MiB (1.3 GB left), so it isn't used. Run one training at a time: two need more than the memory left. Dry runs in `runs/dryrun_train_*`.

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
- If training doesn't fit next to both servers, stop the student server (`scripts/stop_vllm.sh student`) while training, and keep Lightning up. With the task 2.2 settings it fits, so both servers can stay up.
- Extra arguments to a serve script go to `vllm serve` and override its flags (for example a different `--gpu-memory-utilization` for a benchmark in task 4.1).
