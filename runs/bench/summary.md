# Serving benchmarks (task 4.1)

Run `20261018-081524_lora` (2026-10-18T08:15 UTC, commit `346daf8`, with uncommitted changes), built by `bench/run.py`. One H200; Lightning's server was up and idle throughout. Traffic: every model call of the r1 practice runs (`bench/traffic.py`), replayed with `vllm bench serve` as recorded:

| specialist | run | sessions | calls (tool calls, answers, self-checks) | prompt tokens | output tokens |
|---|---|---|---|---|---|
| helpdesk | `r1_helpdesk` | 36 | 147 (57, 45, 45) | 188,834 (run: 188,834) | 8,781 (run: 8,695) |
| ops | `r1_ops` | 36 | 180 (110, 35, 35) | 325,480 (run: 325,480) | 13,204 (run: 13,088) |

## 1. GPU memory: one server with both adapters against one merged server per specialist

| configuration | servers | flags | GPU memory taken | weights | KV and Mamba cache | cache tokens | max concurrency at 32k tokens |
|---|---|---|---|---|---|---|---|
| A: one server: the student with both adapters as LoRA | 1 | 0.15 of the GPU, LoRA | 21,650 MiB | 7.71 GiB | 11.36 GiB | 624,230 | 19.05x |
| B1: two servers, one merged specialist each, A's memory in all | 2 | 0.075 of the GPU + 0.075 of the GPU, merged | 20,627 MiB | 7.47 GiB + 7.47 GiB | 1.52 GiB + 1.52 GiB | 82,739 + 82,739 | 2.52x + 2.52x |
| B2: two servers, one merged specialist each, A's memory per server | 2 | 0.15 of the GPU + 0.15 of the GPU, merged | 43,079 MiB | 7.47 GiB + 7.47 GiB | 11.71 GiB + 11.71 GiB | 643,891 + 643,891 | 19.65x + 19.65x |

## 1 and 2. Throughput and latency at each number of simultaneous sessions

| configuration | sessions | requests (failed) | output tokens per request | output tok/s | total tok/s | requests/s | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms) | end to end p50 / p95 (s) |
|---|---|---|---|---|---|---|---|---|---|
| A | 1 | 80 (0) | 54.2 | 219 | 5,341 | 4.03 | 49 / 55 | 3.8 / 3.8 | 0.21 / 0.42 |
| A | 8 | 327 (0) | 67.2 | 744 | 18,145 | 11.06 | 62 / 117 | 8.4 / 10.3 | 0.49 / 1.26 |
| A | 32 | 327 (0) | 67.2 | 1,239 | 30,221 | 18.43 | 117 / 360 | 21.0 / 25.1 | 1.15 / 3.09 |
| B1 | 1 | 80 (0) | 54.2 | 323 | 7,895 | 5.96 | 30 / 40 | 2.6 / 2.6 | 0.14 / 0.28 |
| B1 | 8 | 327 (0) | 67.2 | 875 | 21,343 | 13.01 | 66 / 119 | 7.6 / 9.3 | 0.45 / 1.14 |
| B1 | 32 | 327 (0) | 67.2 | 1,459 | 35,585 | 21.70 | 370 / 557 | 13.6 / 16.8 | 1.05 / 2.24 |
| B2 | 1 | 80 (0) | 54.2 | 325 | 7,930 | 5.99 | 30 / 37 | 2.6 / 2.6 | 0.14 / 0.29 |
| B2 | 8 | 327 (0) | 67.2 | 861 | 20,991 | 12.80 | 59 / 105 | 7.9 / 9.3 | 0.45 / 1.15 |
| B2 | 32 | 327 (0) | 67.2 | 1,631 | 39,785 | 24.26 | 89 / 383 | 15.8 / 19.6 | 0.91 / 2.24 |

## The merged models against the adapters

Greedy completions (32 tokens) of the first 8 calls of each specialist, identical to configuration A's (BF16 merges can round differently):

| configuration | identical to A |
|---|---|
| B1 | 15/16 |
| B2 | 15/16 |

## Server flags

A, student:

    .venv-vllm/bin/vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f --served-model-name student --port 8011 --max-model-len 32768 --mamba-ssm-cache-dtype float32 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --structured-outputs-config {"backend": "xgrammar", "disable_any_whitespace": true} --gpu-memory-utilization 0.15 --max-num-seqs 256 --enable-lora --max-lora-rank 16 --max-loras 4 --lora-modules helpdesk_r1=adapters/helpdesk_r1 ops_r1=adapters/ops_r1

B1, helpdesk_r1:

    .venv-vllm/bin/vllm serve runs/bench/merged/helpdesk_r1 --served-model-name helpdesk_r1 --port 8012 --max-model-len 32768 --mamba-ssm-cache-dtype float32 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --structured-outputs-config {"backend": "xgrammar", "disable_any_whitespace": true} --gpu-memory-utilization 0.075 --max-num-seqs 16

B1, ops_r1:

    .venv-vllm/bin/vllm serve runs/bench/merged/ops_r1 --served-model-name ops_r1 --port 8013 --max-model-len 32768 --mamba-ssm-cache-dtype float32 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --structured-outputs-config {"backend": "xgrammar", "disable_any_whitespace": true} --gpu-memory-utilization 0.075 --max-num-seqs 16

B2, helpdesk_r1:

    .venv-vllm/bin/vllm serve runs/bench/merged/helpdesk_r1 --served-model-name helpdesk_r1 --port 8012 --max-model-len 32768 --mamba-ssm-cache-dtype float32 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --structured-outputs-config {"backend": "xgrammar", "disable_any_whitespace": true} --gpu-memory-utilization 0.15 --max-num-seqs 256

B2, ops_r1:

    .venv-vllm/bin/vllm serve runs/bench/merged/ops_r1 --served-model-name ops_r1 --port 8013 --max-model-len 32768 --mamba-ssm-cache-dtype float32 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --structured-outputs-config {"backend": "xgrammar", "disable_any_whitespace": true} --gpu-memory-utilization 0.15 --max-num-seqs 256

## Method

- Requests are the recorded prompts sent raw (`--skip-chat-template`, in session order with `--disable-shuffle`) with `--ignore-eos --custom-output-len -1` for each call's recorded output length (the table's output tokens per request are what the servers generated); sampling as `configs/eval.yaml`. Self-checks were structured outputs in the runs; here they are plain completions.
- N sessions are `--max-concurrency` N. With 1 session the two specialists run one after the other on their first 40 calls each; with more, both at once with N/2 sessions each on all their calls, each to its own adapter (A) or server (B). Throughput: both runs' tokens over the summed (one after the other) or longer (together) duration. Two warm-up requests per run.
- GPU memory taken: nvidia-smi's used memory with the configuration's servers up, minus before they started. Weights, cache and max concurrency: the servers' own startup log.

<!-- 4. structured outputs -->
## 4. Structured outputs: valid with and without guided decoding

Run `20261018-090408_structured` (`bench/structured.py`), on the servers as they run. Plans: the planner's request on the 8 multi-API practice cases with both specialists, 3 times each, from Lightning and from the student base. Self-checks: the seatbelt's question after each of the 80 answers in the r1 practice runs, from the specialist's own adapter and from Lightning. Thinking off, sampling as `configs/eval.yaml` but at most 2,048 tokens. JSON: the whole answer parses, as our code parses it; fits: and the JSON schema accepts it. Servers: `scripts/serve_lightning.sh` and `scripts/serve_student.sh` as they run, both with `--structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'`; guided decoding is `response_format` with the request's JSON schema, `strict`.

| request | model | guided decoding | JSON | fits the schema | JSON only in a code fence | cut off at 2,048 tokens | mean output tokens | mean latency (s) |
|---|---|---|---|---|---|---|---|---|
| plan | lightning | on | 24/24 | 24/24 | 0/24 | 0/24 | 197 | 2.45 |
| plan | lightning | off | 22/24 | 8/24 | 0/24 | 2/24 | 403 | 5.22 |
| plan | student | on | 24/24 | 24/24 | 0/24 | 0/24 | 130 | 1.60 |
| plan | student | off | 24/24 | 21/24 | 0/24 | 0/24 | 172 | 2.06 |
| self-check | lightning | on | 80/80 | 80/80 | 0/80 | 0/80 | 55 | 1.66 |
| self-check | lightning | off | 69/80 | 68/80 | 9/80 | 0/80 | 69 | 1.76 |
| self-check | the specialist's adapter | on | 80/80 | 80/80 | 0/80 | 0/80 | 54 | 0.66 |
| self-check | the specialist's adapter | off | 80/80 | 80/80 | 0/80 | 0/80 | 59 | 0.66 |
<!-- /4. structured outputs -->
