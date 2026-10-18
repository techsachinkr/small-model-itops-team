# Commands log

Commands run on the Brev box, in order, each with a comment saying what it was for. Newest at the bottom. Run from the repo root unless noted.

`$SCRATCH` is a session scratch directory outside the repo, used for throwaway scripts and downloads. It doesn't survive the session; anything worth keeping gets moved into the repo.

## 2026-10-17: task 0.2 Set up

### Research: Lightning's model card, vLLM version, student shortlist

```bash
# Lightning: HF metadata (gated?, license, safetensors parameter count) and the raw model card
curl -s https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
curl -s https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/README.md

# vLLM release dates on PyPI, to check rule 1 (released on or before 2026-10-03)
curl -s https://pypi.org/pypi/vllm/json

# vLLM's pinned dependencies (both 0.27.1 and 0.30.0 pin torch==2.13.0; our train group has 2.14.1)
curl -s https://pypi.org/pypi/vllm/0.27.1/json
curl -s https://pypi.org/pypi/vllm/0.30.0/json

# vLLM release notes v0.25.0 to v0.30.0, to see what each added for Nemotron-H, DSpark and MTP
curl -s https://api.github.com/repos/vllm-project/vllm/releases/tags/v0.27.1

# vLLM v0.27.1 source: LoRA support in NemotronH, Mamba projections, built-in tool and reasoning parsers
curl -s https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/vllm/model_executor/models/nemotron_h.py
curl -s https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/vllm/model_executor/layers/mamba/mamba_mixer2.py
curl -s https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/vllm/tool_parsers/__init__.py
curl -s https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/vllm/reasoning/__init__.py
curl -s https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/vllm/parser/nemotron_v3.py
curl -s https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/docs/features/tool_calling.md

# Student candidates: list NVIDIA's small Nemotron models, then metadata and cards for the shortlist
curl -s "https://huggingface.co/api/models?author=nvidia&search=Nemotron&limit=500&full=true"
curl -s https://huggingface.co/api/models/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
curl -s https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/raw/main/README.md
curl -s https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507
curl -s https://huggingface.co/api/models/ibm-granite/granite-4.0-micro

# Nano 4B's optional reasoning-parser plugin, read only (not used: built-in nemotron_v3 covers it)
curl -sL -o $SCRATCH/dl_nano/nano_v3_reasoning_parser.py \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/resolve/main/nano_v3_reasoning_parser.py

# Chat templates: Lightning and Nano 4B both emit <tool_call><function=...><parameter=...> (qwen3_coder format)
curl -sL https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/raw/main/chat_template.jinja
curl -sL https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/raw/main/chat_template.jinja
```

### vLLM in its own venv

```bash
# Separate venv for vLLM: it pins torch 2.13.0, and the project's train group uses 2.14.1
uv venv .venv-vllm --python 3.13

# The version pinned by Lightning's card, with the same rule 1 date cutoff as pyproject.toml
uv pip install --python .venv-vllm/bin/python vllm==0.27.1 --exclude-newer 2026-10-03T00:00:00Z

# Check: printed "vllm 0.27.1 torch 2.13.0+cu130 cuda 13.0 True"
.venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available())"
```

### Student base download

```bash
# Nano 4B at the commit pinned in configs/models.yaml (~8 GB, into the HF cache under ~/.cache)
.venv/bin/hf download nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f
```

### LoRA check: do PEFT adapters for Nano 4B load and apply correctly in vLLM?

The two scripts are now in the repo as `scripts/checks/lora_make_adapters.py` and `scripts/checks/lora_check_vllm.py`. The commands below ran them from `$SCRATCH` under their original names, `make_adapters.py` and `check_vllm.py`. To rerun, use the repo paths and any `--out` directory.

Three untrained adapters (r=16, alpha=32): `attn_mlp_zero` (attention + MLP, PEFT's default zero init, so it must match the base model exactly), `attn_mlp_rand` (same targets, small random weights, so it must differ from base and match HF+PEFT), and `mamba_rand` (Mamba `in_proj` only, random weights).

```bash
# Is NemotronH native in our transformers (no remote code needed), and which module names exist?
# Result: native in transformers 5.18.0; mamba_ssm and causal_conv1d kernels are not installed,
# so HF runs and trains Mamba layers on the slower plain-PyTorch path.
.venv/bin/python -c "import importlib; importlib.import_module('transformers.models.nemotron_h.modeling_nemotron_h')"

# Build the adapters with PEFT and record HF greedy outputs (32 tokens) for base and each adapter.
# First attempt targeted Mamba in_proj + out_proj: PEFT refuses out_proj and conv1d on nemotron_h,
# so mamba_rand targets in_proj only.
.venv/bin/python $SCRATCH/lora_check/make_adapters.py \
  --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f \
  --out $SCRATCH/lora_check/adapters

# Serve the student with the configs/models.yaml parsers and LoRA enabled. Runtime adapter loading
# is on so each adapter's load result can be checked separately. Low memory share: test only.
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True .venv-vllm/bin/vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f --served-model-name nano4b --port 8001 \
  --gpu-memory-utilization 0.25 --max-model-len 8192 \
  --enable-lora --max-lora-rank 16 --max-loras 4 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
  --mamba-ssm-cache-dtype float32 > $SCRATCH/lora_check/vllm_serve.log 2>&1

# Wait until the server answers /v1/models (or report the log if it dies)
until curl -sf localhost:8001/v1/models >/dev/null; do sleep 1; done
```

That first launch failed: `FileNotFoundError: 'ninja'`. FlashInfer JIT-compiles kernels at startup and runs `ninja` through PATH. Ninja is installed in `.venv-vllm/bin`, but calling `.venv-vllm/bin/vllm` directly doesn't put that directory on PATH. `nvcc` was already on PATH (system CUDA 13.0 at `/usr/local/cuda`). Fix: put the venv's bin first on PATH. The serve scripts must do the same.

```bash
# Where are ninja and nvcc? ninja: only in .venv-vllm/bin; nvcc: /usr/local/cuda/bin
ls .venv-vllm/bin/ninja; which ninja nvcc

# Relaunch with the vLLM venv on PATH (same flags as above)
PATH=$PWD/.venv-vllm/bin:$PATH VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --revision dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f --served-model-name nano4b --port 8001 \
  --gpu-memory-utilization 0.25 --max-model-len 8192 \
  --enable-lora --max-lora-rank 16 --max-loras 4 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
  --mamba-ssm-cache-dtype float32 > $SCRATCH/lora_check/vllm_serve.log 2>&1
```

Ready after about 2 minutes (startup includes torch.compile and CUDA graph capture). From its startup log: `trust_remote_code=False`, and `enable_prefix_caching=False` by default for this hybrid model (relevant to task 4.1).

```bash
# Load the three adapters at runtime (POST /v1/load_lora_adapter); greedy-complete the same prompt
# token IDs with base and each adapter and compare with the HF+PEFT outputs; then send one
# get_weather tool-call request with thinking off and one with thinking on, to test both parsers.
.venv/bin/python $SCRATCH/lora_check/check_vllm.py --adapters $SCRATCH/lora_check/adapters
```

Result:

| Model | Loads in vLLM | vLLM matches HF+PEFT (greedy tokens) | Differs from base |
|---|---|---|---|
| base | – | 32/32 | – |
| `attn_mlp_zero` | yes | 32/32 | no (expected: zero init) |
| `attn_mlp_rand` | yes | 32/32 | yes |
| `mamba_rand` (`in_proj`) | yes | 32/32 | yes |

Tool call, thinking off: parsed `get_weather({"city": "Raleigh"})`, empty content, no reasoning. Thinking on: the same parsed call, with 147 characters split into the reasoning field. So the built-in `nemotron_v3` and `qwen3_coder` parsers work for Nano 4B. GPU memory while serving: 36.4 GB, which is the 0.25 reservation, not what the model needs.

```bash
# Stop the test server and free the GPU. Note: `pkill -f "vllm serve ..."` also matched (and killed)
# the shell running it, because the pattern was in that shell's own command line. The server did stop.
pkill -f "vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B"

# Check nothing is left: printed "no vllm processes" and "0 MiB, 143771 MiB"
pgrep -af vllm; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

```bash
# Keep the LoRA check scripts in the repo (the scratch directory doesn't survive the session)
mkdir -p scripts/checks
cp $SCRATCH/lora_check/make_adapters.py scripts/checks/lora_make_adapters.py
cp $SCRATCH/lora_check/check_vllm.py scripts/checks/lora_check_vllm.py
# Check both still compile
.venv/bin/python -m py_compile scripts/checks/lora_make_adapters.py scripts/checks/lora_check_vllm.py
```

### Lightning download (ahead of 0.2 Serving)

```bash
# Free disk before the download: printed 248 GB free on / (the HF cache at ~/.cache is on the same disk)
df -h /

# Lightning BF16 at the commit pinned in configs/models.yaml (~63 GB, into the HF cache). Run in the background.
.venv/bin/hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 > $SCRATCH/lightning_download.log 2>&1

# Check: cache size for the model, and free disk afterwards
du -sh ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
df -h /
```

Result: finished in about 10 minutes. All 14 shards are in the snapshot, with no `.incomplete` files left. Free disk went from 248 GB to 186 GB. The cache folder itself reports about 1 MB to plain `du`, because the snapshot links point outside it; `du -shL` on the snapshot gives the real size, 62 GB.

```bash
# Check each shard is complete: its size must equal 8 + header length + the last tensor's end offset.
# Printed "shards ok: True | tensor bytes 65826538624 | params 32913266240". The index's own
# total_size (65842365568) is slightly larger than the files hold; it's stale metadata, not missing data.
python3 -I -c "<header check; see the task 0.2 session>" \
  /ephemeral/cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/a9904d24bcc1d289a1950fa9d2b978c47cf903b9
du -shL ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/a9904d24bcc1d289a1950fa9d2b978c47cf903b9
```

### Event requirements (0.2 checklist item)

You copied the event page to `event_req/NVIDIA x Red Hat Small Models Hack.pdf`. Its text has links to the Submission Checklist and Demo Video Instructions but not their contents, and the bounty details are images. The results went into PLAN.md, "Event requirements".

```bash
# The Read tool can't render PDFs here (no poppler), and no PDF library is installed. Run pypdf in a
# throwaway uvx environment (it doesn't touch the project's dependencies) to extract the text.
uvx --from pypdf==5.4.0 python -I -c "<write each page's extract_text() to a file>" \
  "event_req/NVIDIA x Red Hat Small Models Hack.pdf" $SCRATCH/event_req.txt

# Extract link targets (the Airtable form and the Notion subpages) and embedded images. The bounty
# slides on pages 5–6 are images, which were then viewed directly.
uvx --from pypdf==5.4.0 --with pillow==11.1.0 python -I -c "<print /Annots URIs; save page.images>" \
  "event_req/NVIDIA x Red Hat Small Models Hack.pdf" $SCRATCH/event_req_img

# The Notion pages render with JavaScript, so a plain fetch returns nothing. Notion's public page-data
# endpoint returns each page's blocks as JSON (repeat with the Demo Video Instructions page ID
# 087f567d-17cc-8239-839b-011a10c5c0d3, saved as video.json).
curl -s -X POST https://concrete-panther-c83.notion.site/api/v3/loadCachedPageChunk \
  -H "Content-Type: application/json" \
  -d '{"page":{"id":"9e5f567d-17cc-83c0-9dc7-0160ca887731"},"limit":200,"cursor":{"stack":[]},"chunkNumber":0,"verticalColumns":false}' \
  -o $SCRATCH/notion/checklist.json

# Print the blocks as indented text in page order (helper kept in the repo as scripts/tools/notion_page_text.py)
python3 -I scripts/tools/notion_page_text.py $SCRATCH/notion/checklist.json 9e5f567d-17cc-83c0-9dc7-0160ca887731
python3 -I scripts/tools/notion_page_text.py $SCRATCH/notion/video.json 087f567d-17cc-8239-839b-011a10c5c0d3
```

## 2026-10-17: task 0.2 Serving, the LLM client and replay mode

### Checks before writing the serve scripts

```bash
# Every flag in Lightning's card recipe exists in the installed vLLM 0.27.1 (mamba backend, stochastic
# rounding, philox rounds, async scheduling); --lora-modules takes name=path; --gpu-memory-utilization is
# a per-instance share of the whole GPU, so two servers' shares add up.
PATH=$PWD/.venv-vllm/bin:$PATH vllm serve --help=all > $SCRATCH/vllm_help.txt

# Both chat templates default enable_thinking to true; Lightning's card validates 256K context on one
# H100 (we use 32K); Nano 4B's card uses --mamba-ssm-cache-dtype float32 (as in the LoRA check).
grep -n "enable_thinking" ~/.cache/huggingface/hub/models--nvidia--*/snapshots/*/chat_template.jinja
grep -n -A22 "vllm serve" ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/snapshots/*/README.md

# vLLM returns reasoning in the message field `reasoning`, but both chat templates read
# `reasoning_content` from history. For task 1.2: rename it when the tool loop sends history back.
grep -n "class ChatMessage(" -A14 .venv-vllm/lib/python3.13/site-packages/vllm/entrypoints/openai/chat_completion/protocol.py
```

### Serve Lightning, write the client

```bash
# Start Lightning in the background (runs/vllm/lightning.{pid,log}); the script waits for /v1/models
chmod +x scripts/serve_lightning.sh scripts/serve_student.sh scripts/stop_vllm.sh
scripts/serve_lightning.sh

# common/llm.py offline tests (stub client, no GPU): 17 passed (11 schema + 6 llm)
.venv/bin/python -m pytest -q

# Flags for the price benchmarks (step 5)
PATH=$PWD/.venv-vllm/bin:$PATH vllm bench serve --help=all > $SCRATCH/bench_help.txt
```

Lightning ready after 135 s. From its log: weights 58.92 GiB, KV cache 14.98 GiB (513,440 tokens; 15.67x
concurrency at the full 32K), CUDA graphs 0.43 GiB. `nvidia-smi`: 78,239 MiB used with Lightning alone.

```bash
# Start the student next to Lightning (background; runs/vllm/student.{pid,log})
scripts/serve_student.sh

# Smoke test, Lightning, live, thinking on (card sampling 1.0/0.95): 20/20 parsed, all 20 with reasoning,
# unit passed through when asked. runs/smoke_toolcall/lightning_live_20261017-170439/
.venv/bin/python scripts/smoke_toolcall.py --model lightning

# Structured output through common/llm.py on Lightning, thinking on: the schema constrains only the
# answer (finish "stop", reasoning field filled, content parses as {"complete": true, "reason": ...}).
.venv/bin/python - <<'PY'  # llm.chat("lightning", msgs, json_schema={complete: bool, reason: str}, ...)
PY
```

The first student launch failed: `ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (643)`.
vLLM's default of 1,024 sequences needs one Mamba state block each, and the 0.15 share holds 643 (weights
7.71 GiB, KV cache 9.58 GiB). Lightning was unaffected. Fix: `--max-num-seqs 256` in serve_student.sh.

```bash
# Find the root cause in the student log (the script's tail showed only the API server's traceback)
grep -nE "Error|memory|KV cache|mamba" runs/vllm/student.log

# Relaunch the student with --max-num-seqs 256
scripts/serve_student.sh
```

Student ready after 40 s. Both servers on the GPU together (`nvidia-smi`): Lightning 78,230 MiB, student
21,644 MiB, total 99,889 of 143,771 MiB. Student: weights 7.71 GiB, KV cache 11.36 GiB (624,230 tokens).
From the logs: prefix caching on for Lightning, off for the student. Details in configs/gpu.md.

```bash
# Both servers loaded; per-process memory and served model names
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
curl -s localhost:8000/v1/models; curl -s localhost:8001/v1/models

# Smoke test, student, live, thinking on (card sampling 0.6/0.95): 20/20 parsed, all 20 with reasoning.
# runs/smoke_toolcall/student_live_20261017-170725/
.venv/bin/python scripts/smoke_toolcall.py --model student

# Structured output on the student, thinking on: finish "stop", content parses against the schema
.venv/bin/python - <<'PY'  # llm.chat("student", msgs, json_schema={complete: bool, reason: str}, ...)
PY
```

### Prices (configs/eval.yaml)

GPU price: $6.53/hour (Brev H200, from the user). Throughput is measured with vLLM's own benchmark
client against each server while the other is idle, with the same settings for both models.

```bash
# Per model: a prefill-dominated run (prompt price) and a decode-dominated run (completion price).
# Random token prompts, 256 requests, concurrency 64, seed 0. The tokenizer comes from the pinned snapshot.
PATH=$PWD/.venv-vllm/bin:$PATH HF_HUB_OFFLINE=1 vllm bench serve --backend vllm \
  --base-url http://localhost:8000 --model lightning --tokenizer <lightning snapshot> \
  --dataset-name random --random-input-len 2048 --random-output-len 1 \
  --num-prompts 256 --max-concurrency 64 --ignore-eos --seed 0 \
  --save-result --result-dir runs/prices --result-filename lightning_prefill.json
# ... the same with --random-input-len 128 --random-output-len 512 -> lightning_decode.json,
# and both again with --base-url http://localhost:8001 --model student -> student_{prefill,decode}.json

# Prices from the saved result files: price_per_1m = 6.53 / (tok_per_s * 3600 / 1e6)
.venv/bin/python -I - runs/prices <<'PY'  # prompt tok/s = total_input_tokens / duration (prefill run);
PY                                        # completion tok/s = output_throughput (decode run)
```

| Model | Prompt tok/s | Completion tok/s | $/1M prompt | $/1M completion | Failed requests |
|---|---|---|---|---|---|
| lightning | 56,609 | 3,949 | 0.0320 | 0.4594 | 0/512 |
| student | 77,991 | 9,430 | 0.0233 | 0.1924 | 0/512 |

### Replay check (Done when)

```bash
# Record a smoke run for each model into runs/cache/ (40 entries): both 20/20
LLM_MODE=record .venv/bin/python scripts/smoke_toolcall.py --model lightning
LLM_MODE=record .venv/bin/python scripts/smoke_toolcall.py --model student

# Stop both servers: printed "lightning: stopped", "student: stopped", "0 MiB, 143771 MiB";
# no vllm processes left, nothing answers on 8000 or 8001
scripts/stop_vllm.sh
pgrep -af "vllm serve"; curl -s -m 2 localhost:8000/v1/models; curl -s -m 2 localhost:8001/v1/models

# Replay with no server: both 20/20; 20/20 responses identical to the recording for each model
# (message and token counts), all marked replayed; an unrecorded request raises ReplayMiss
LLM_MODE=replay .venv/bin/python scripts/smoke_toolcall.py --model lightning
LLM_MODE=replay .venv/bin/python scripts/smoke_toolcall.py --model student
.venv/bin/python -I - runs/smoke_toolcall <<'PY'  # compare latest record vs replay results.jsonl
PY

# Final offline tests
.venv/bin/python -m pytest -q
```

### Verify (the "Done when" check, run again from a clean start): passed

```bash
# Start both servers: Lightning ready after 45 s, student after 34 s (compile caches warm)
scripts/serve_lightning.sh && scripts/serve_student.sh

# (1) Both on the GPU together: Lightning 78,994 MiB, student 21,644 MiB, total 100,653 of 143,771 MiB
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
curl -s localhost:8000/v1/models; curl -s localhost:8001/v1/models

# (2) Smoke test, live with recording: lightning 20/20, student 20/20 (identical to the first run:
# seeded sampling)
LLM_MODE=record .venv/bin/python scripts/smoke_toolcall.py --model lightning
LLM_MODE=record .venv/bin/python scripts/smoke_toolcall.py --model student

# (3) Stop vLLM (0 MiB used, no processes, nothing on 8000/8001), then replay: both 20/20, and 20/20
# responses identical to the recording for each model
scripts/stop_vllm.sh
LLM_MODE=replay .venv/bin/python scripts/smoke_toolcall.py --model lightning
LLM_MODE=replay .venv/bin/python scripts/smoke_toolcall.py --model student
.venv/bin/python -I - runs/smoke_toolcall <<'PY'  # compare latest record vs replay results.jsonl
PY
```

```bash
# Wrap up: commit task 0.2 on main (event_req/ left out: untracked, not part of this task)
git add CLAUDE.md COMMANDS_LOG.md PLAN.md common/llm.py configs/ scripts/ tests/test_llm.py
git commit
```

## 2026-10-17: task 1.1 The NeMo Gym environment

### Research: NeMo Gym 0.6.0

The installed wheel has no `docs/` folder. Read the v0.6.0 docs online (Stateful Environment, Execution and State
Match, Single-Step Environment) and the installed source under `.venv/lib/python3.13/site-packages/`:
`resources_servers/example_session_state_mgmt`, `resources_servers/workplace_assistant`,
`responses_api_agents/simple_agent/app.py`, `nemo_gym/base_resources_server.py`, `nemo_gym/cli/env.py`,
`nemo_gym/cli/setup_command.py`.

### Loghub sample

```bash
# Pinned commit of github.com/logpai/loghub, and its license (the notice must go with every copy)
curl -s https://api.github.com/repos/logpai/loghub/commits/master          # dd61d09..., 2025-06-13
curl -s https://raw.githubusercontent.com/logpai/loghub/dd61d0952749ee7963bde24220d1be5ede023033/LICENSE
# Download three structured 2k samples + LICENSE into a new folder (1.6 MB)
mkdir data/open/loghub && cd data/open/loghub && REF=dd61d0952749ee7963bde24220d1be5ede023033 && \
  for f in LICENSE OpenStack/OpenStack_2k.log_structured.csv Zookeeper/Zookeeper_2k.log_structured.csv \
           Hadoop/Hadoop_2k.log_structured.csv; do
    curl -sSfL -o "$(basename $f)" "https://raw.githubusercontent.com/logpai/loghub/$REF/$f"; done
sha256sum *   # Hadoop e88e5bf9..., LICENSE 9ffa6ae2..., OpenStack ecc88e79..., Zookeeper e4a450c6...
```

### Build and test

```bash
# Fixtures: logs.json (30 images) and 5 incident variants, from default.json + Loghub, seed 0
.venv/bin/python -m gym_env.fixtures.make_fixtures --config configs/fixtures.yaml
# Hand-made example cases -> Gym rows (7 rows)
.venv/bin/python -m gym_env.tasks --cases resources_servers/itops/data/example_cases.jsonl \
  --output resources_servers/itops/data/example.jsonl
# Tests: 74 passed. Then three deliberate mutations, each caught and reverted: exact summaries (1 failed),
# no change_ticket rule (5 failed), unseeded chaos (1 failed)
.venv/bin/python -m pytest -q
```

### Verify (the "Done when" check): passed

```bash
.venv/bin/python -m pytest -q                       # 74 passed
scripts/serve_lightning.sh                          # ready after 44 s
# First try, end to end in one command: refused ("-i/--input is not supported when serving end-to-end;
# add --no-serve"), so start the servers separately. UV_EXCLUDE_NEWER keeps Gym's per-server venvs
# within rule 1; they go to runs/gym_venvs/ (1.4 GB, gitignored).
UV_EXCLUDE_NEWER=2026-10-03T00:00:00Z .venv/bin/gym env start --resources-server itops --model-type vllm_model \
  --model lightning --model-url http://localhost:8000/v1 --model-api-key EMPTY +uv_venv_dir=runs/gym_venvs \
  > runs/gym_smoke/gym_env.log 2>&1 &
# 7/7 rollouts collected; mean reward 0.857: 6 of 7 passed. ex-multi-001 (the demo request) failed on
# tickets.HD-1042.category: expected 'incident', got 'software' (ticket first, HD-1042 passed to the prod
# rollback, revision 8: all correct)
.venv/bin/gym eval run --no-serve --agent itops_simple_agent \
  --input resources_servers/itops/data/example.jsonl --output runs/gym_smoke/rollouts.jsonl
# Stop the Gym servers (Ctrl+C / kill the gym env start process), then Lightning; GPU back to 0 MiB
scripts/stop_vllm.sh lightning
```

### Category definitions in create_ticket's schema, and a rerun

```bash
# After adding one-line definitions per category to create_ticket's `category` argument: regenerate the
# Gym rows (tool schemas are embedded in them) and test: 74 passed
.venv/bin/python -m gym_env.tasks --cases resources_servers/itops/data/example_cases.jsonl \
  --output resources_servers/itops/data/example.jsonl
.venv/bin/python -m pytest -q
# Rerun with Lightning, 3 rollouts per task: 19/21. ex-multi-001 2/3 (categories: incident, software,
# incident; was software in the single earlier run). ex-ro-001 2/3: the failure answered
# "Status: Open", and answer_must_contain "open" is a case-sensitive substring check, so a correct answer failed.
scripts/serve_lightning.sh                          # ready after 45 s
UV_EXCLUDE_NEWER=2026-10-03T00:00:00Z .venv/bin/gym env start --resources-server itops --model-type vllm_model \
  --model lightning --model-url http://localhost:8000/v1 --model-api-key EMPTY +uv_venv_dir=runs/gym_venvs \
  > runs/gym_smoke/gym_env_2.log 2>&1 &
.venv/bin/gym eval run --no-serve --agent itops_simple_agent --input resources_servers/itops/data/example.jsonl \
  --output runs/gym_smoke/rollouts_categories.jsonl --num-repeats 3
scripts/stop_vllm.sh lightning                      # after stopping the Gym servers; GPU back to 0 MiB
```

### Keep event_req/ out of the repo

```bash
# Commit a8316f1 added event_req/; untrack it (the PDF stays on disk) and gitignore the folder
git rm -r --cached event_req/
printf '\n# Event materials, kept locally only\nevent_req/\n' >> .gitignore
```

### answer_must_contain ignores case

```bash
# verify.py now matches answer_must_contain case-insensitively (casefold). Tests: 75 passed. Re-scoring the
# rerun's three ex-ro-001 answers offline (read-only case, so the final state is the fixture): all pass,
# including the "Status: Open" one that failed before
.venv/bin/python -m pytest -q
```

## 2026-10-17: task 1.2 Seatbelt, tool loop and the baseline

Decisions (with the user): temperature 0.6, top_p 0.95, thinking on, for every eval setup (student card's
tool-calling setting; Lightning is only the reference row); `jsonschema` for the seatbelt's argument check;
max_steps 10 model turns per step; the self-check uses the step's own model and counts toward its tokens.

### Research: Gym agents and the vLLM model server

Read the installed source: `responses_api_agents/simple_agent/app.py`, `nemo_gym/base_responses_api_agent.py`,
`responses_api_models/vllm_model/app.py`, `nemo_gym/responses_converter.py`, `nemo_gym/cli/env.py`
(`_resolve_server_dir`), `nemo_gym/cli/setup_command.py`. Findings: Gym finds our own agent at
`responses_api_agents/<name>/` in the repo (cwd is searched before the install root); the model server also
serves `/v1/chat/completions`, returning reasoning as `<think>...</think>` in the content and turning it back
into `reasoning_content` when the message is sent again; `ResponsesConverter` converts a row's Responses
input and tools to chat format and a chat conversation back to Responses output items.

### Dependency

```bash
# Add jsonschema (already in uv.lock as a nemo-gym dependency, 4.26.0) to pyproject.toml
uv lock
uv sync                       # mistake: without --group train this uninstalled TRL, PEFT, torch...
uv sync --group train         # ...so reinstall the train group (same locked versions)
```

### Build and test

```bash
.venv/bin/python -m pytest -q tests/test_seatbelt.py   # 11 passed, 1 failed: my test expected 404 for
                                                       # not_found; the env returns 400. Fixed the test
.venv/bin/python -m pytest -q                          # 87 passed
```

### Smoke: the itops agent in Gym, Lightning, the 7 example rows

```bash
scripts/serve_lightning.sh                            # ready after 44 s
scripts/serve_student.sh                              # ready after 34 s
UV_EXCLUDE_NEWER=2026-10-03T00:00:00Z .venv/bin/gym env start --config resources_servers/itops/configs/itops.yaml \
  --config gym_configs/eval.yaml --config gym_configs/lightning.yaml +uv_venv_dir=runs/gym_venvs \
  > runs/gym_smoke_agent/gym_env.log 2>&1             # 4/4 servers ready (builds the itops_agent venv)
.venv/bin/gym eval run --no-serve --agent itops_agent --input resources_servers/itops/data/example.jsonl \
  --output runs/gym_smoke_agent/rollouts.jsonl
# 7/7 collected; 6 passed. Two self-check problems:
# - ex-mi-001 (missing_info): Lightning correctly asked for the deployment and change ticket, but its
#   self-check said complete=false ("missing deployment name") three times, so the seatbelt gave up.
# - ex-ro-001: the self-check answer stopped at '{"complete": true' and used ~8k completion tokens:
#   whitespace generated until max_tokens.
```

Fixes: compact JSON on both vLLM servers (`--structured-outputs-config '{"backend": "xgrammar",
"disable_any_whitespace": true}'`; `auto` is rejected with that option; vLLM's xgrammar backend reads it only
from the server config, not per request), and a clearer self-check prompt.

```bash
scripts/stop_vllm.sh && scripts/serve_lightning.sh && scripts/serve_student.sh   # 45 s, 34 s
# Rerun, 3 rollouts per row (servers restarted, so the new prompt is loaded): 19/21 with reward 1.
# All self-check answers parse now, with normal token counts. ex-multi-001 2/3 (category software
# vs incident, as in 1.1). ex-mi-001 2/3, and 2 of its 3 attempts still had complete=false self-checks
# with correct replies.
UV_EXCLUDE_NEWER=2026-10-03T00:00:00Z .venv/bin/gym env start ... > runs/gym_smoke_agent/gym_env_2.log 2>&1 &
.venv/bin/gym eval run --no-serve --agent itops_agent --input resources_servers/itops/data/example.jsonl \
  --output runs/gym_smoke_agent/rollouts_2.jsonl --num-repeats 3
# Stopping it: SIGINT to the gym process alone doesn't stop it; SIGINT to its process group (like Ctrl+C)
# does. (A first `pkill -f "gym env start"` killed my own shell, whose command line matched.)
kill -INT -- -<pgid>
```

Self-check prompt v3: the three cases spelled out (done / asked for missing information / declined), and
`reason` before `complete` in the schema so the model names the case before deciding.

### Practice cases (drafts, for developer review)

```bash
# 12 helpdesk + 12 ops cases in data/practice/. Checked: every case validates, gold calls all succeed, a gold
# run passes the verifier, an idle run fails exactly the single cases, and each read_only answer string
# appears in the gold tool outputs. 24/24 ok.
```

### Smoke: eval/run_eval.py end to end (on the unreviewed drafts; not the baseline)

```bash
.venv/bin/python -m eval.run_eval --setup base_seatbelt --api ops --split practice --run-id smoke_base_seatbelt_ops
# 36/36 attempts collected, servers stopped cleanly (port 11000 free, no raylet left).
# base_seatbelt on ops: 12/12 cases. No invalid calls or tool errors at all. 4 of the student's 42
# self-check answers were empty: it wrote the answer inside a reasoning block it never closed, so the
# schema never applied. pr-ops-10 attempt 3 gave up on those (2/3 still passes).
```

Fix: the self-check runs with thinking off (Gym: `metadata.chat_template_kwargs`, which Gym's model server
merges into the request; vLLM ignores the extra `metadata` field). Tests: 96 passed.

```bash
.venv/bin/python -m pytest -q                                                    # 96 passed
.venv/bin/python -m eval.run_eval --setup base_seatbelt --api helpdesk --split practice \
  --run-id smoke_base_seatbelt_helpdesk
# base_seatbelt on helpdesk: 11/12 cases; all 36 self-checks parsed. The failure, pr-hd-05 0/3, was my
# draft's fault: the student closed the ticket and added a closing note each time; the gold had no note and
# the request didn't say. Reworded to ask for the note, and added it to the gold.
```

### Harder practice cases (the drafts scored at or above the plan's 11/12 bar)

```bash
# v2: 24 new drafts (two-action requests, reopen instead of new ticket, inferred delivery/priority, incident vs
# hardware, joins across lookups, values that need a lookup first, revision choice from change causes,
# resolved change ticket, missing namespace). New test tests/test_practice_cases.py runs the same checks as
# before on every practice case: 121 passed.
.venv/bin/python -m pytest -q
for api in helpdesk ops; do .venv/bin/python -m eval.run_eval --setup base_seatbelt --api $api --split practice \
  --run-id smoke_v2_base_seatbelt_$api; done
# base_seatbelt: helpdesk 10/12, ops 11/12. Every failure checked by hand: all genuine model mistakes
# (incident filed as software; new ticket instead of updating the existing one; ticket not resolved; device
# picked without asking; note added instead of explaining there's no delete tool; prod rolled back to 23
# instead of 22, once via a detour 23 -> 22, which the change log correctly fails).

# v3: replaced 4 easy ops cases (find the unhealthy deployment and roll it back; scale up every staging
# deployment below 2 replicas; roll back to "whatever was running last Friday"; list every crash-looping pod)
# and 2 easy helpdesk cases (ticket for another employee; reset for "Hannah Weber" with no employee ID).
for api in helpdesk ops; do .venv/bin/python -m eval.run_eval --setup base_seatbelt --api $api --split practice \
  --run-id smoke_v3_base_seatbelt_$api; done
# base_seatbelt: helpdesk 10/12, ops 11/12. 0/3 every time: pr-hd-04 (infer sms), pr-hd-11 (ask which
# device), pr-ops-04 (revision from change causes). Several cases flip between runs (pr-hd-06 1/3 -> 3/3,
# pr-hd-11 2/3 -> 0/3). Stopped iterating: with thinking on, the student handles multi-step lookups,
# filters, dates and the change-ticket rule; harder cases now mean fitting the practice set to its few
# weak spots. Decision for the user.
```

# Also wrote PROJECT_OVERVIEW.md (what the system does, for readers new to the project).

### Thinking on vs off, every setup (on the unreviewed v3 drafts)

```bash
# thinking became one shared eval setting (configs/eval.yaml `thinking`, still true); the thinking-off runs use
# runs/experiments/eval_thinking_off.yaml, identical except `thinking: false`. Tests: 121 passed.
for spec in off:base_raw off:base_seatbelt off:lightning on:base_raw on:lightning; do ...
  .venv/bin/python -m eval.run_eval --setup $setup --api $api --split practice --config $cfg \
    --run-id think_${mode}_${setup}_${api}; done      # all 10 runs: 36/36 attempts collected
```

| setup | api | thinking on | thinking off |
|---|---|---|---|
| base_raw | helpdesk | 10/12 | 5/12 |
| base_raw | ops | 11/12 | 9/12 |
| base_seatbelt | helpdesk | 10/12 (smoke_v3) | 8/12 |
| base_seatbelt | ops | 11/12 (smoke_v3) | 8/12 |
| lightning | helpdesk | 9/12 | 7/12 |
| lightning | ops | 8/12 | 6/12 |

Thinking off: about 3x fewer completion tokens per attempt and about half the latency, for every setup.
Lightning below the student is the self-check's doing: with thinking on, the verifier passed 33/36 (helpdesk)
and 30/36 (ops) of Lightning's attempts, and the self-check discarded 7 correct attempts on each. Lightning
answers complete=false on ask-the-user cases even when its reason says it asked the user. Across the 8
seatbelt runs, the self-check discarded 24 verifier-correct attempts (22 Lightning's) and almost never led to
a missing call being made. The schema check and retries do work with thinking off (student ops: 4 invalid
calls, 9 tool errors sent back).

Also: wrote eval/report.py (results table from run folders) and runs/practice_review.md (review sheet).

### Thinking off for every setup, and the self-check reworded (user's decision)

configs/eval.yaml `thinking: false`. Self-check question changed from "is the task complete?" to "is there a
tool call you still need to make now?" (`{reason, tool_call_needed}`; the trace still logs `complete`).
CLAUDE.md "Seatbelt" updated. Tests: 121 passed.

```bash
for api in helpdesk ops; do .venv/bin/python -m eval.run_eval --setup lightning --api $api --split practice \
  --run-id selfcheck_v4_lightning_$api; done
# lightning (thinking off): helpdesk 9/12, ops 9/12 (was 7/12 and 6/12 with the old self-check). Correct
# attempts the self-check discarded: helpdesk 2 (both pr-hd-10, user with no tickets), ops 0 (was 7 and 7 with
# thinking on, 1 and 8 with thinking off). Remaining failures are genuine: reset the current user's password
# instead of asking for Hannah's ID; scaled prod payments to 2 on a guess, then to 5 (two prod changes);
# answered two read-only questions without any tool call; created a ticket without asking which device.
```

### Evaluator review of the practice cases, and fixes (user approved)

An independent subagent (evaluator persona, no access to data/test/) reviewed all 24 cases by scoring alternative
call sequences against the env and reading the rollouts. No gold was wrong. I re-checked its main claims in the
run data: pr-ops-11's 6/6 passes were all prod guesses saved only because the made-up change ticket was rejected;
in `default` every prod deployment still runs the revision it ran yesterday, so pr-ops-12 had nothing to roll back;
pr-ops-07 was read as "search" the verb (base looked up payments 2/3); pr-hd-08's base failures all filtered
status=open on an in_progress ticket. Correction to my earlier note: pr-hd-10's two "discarded" Lightning attempts
were max-steps loops on list_tickets, not the self-check; but on pr-hd-09 the self-check did push a correct
question into resetting the current user's password.

Applied:
- verify.py: tickets the gold creates or updates need at least the gold's number of notes (others stay exact);
  new optional Case field answer_must_not_contain (schema validates it isn't part of a required string; the
  server's verify response gains forbidden_in_answer). CLAUDE.md "Data formats" and "Scoring" updated.
- Cases: pr-hd-02 "in this order"; pr-hd-08 keeps "open" (users call in_progress tickets open; the trap is now in
  its stated intent), must not contain the other device; pr-ops-07 "revisions of the search deployment", must not
  contain 0.9.5; pr-ops-08 must not contain any healthy pod name; pr-ops-11 gives change ticket HD-1041 so a guess
  in either namespace changes state; pr-ops-12 is now a single case on incident_prod_checkout with HD-1041 (read
  every history, roll back checkout 14 -> 13).
- Tests for each rule and each fixed case's intent (right behaviour passes, the wrong one the review found fails).

```bash
.venv/bin/python -m pytest -q           # 129 passed
```

Not applied (recorded for later): a check that prod change tickets come from the request or a tool result (goes
beyond state match, rule 6); unordered matching of new tickets; a new fixture with several tickets per user and
real change tickets (a new fixture, not edits to `default`, so the test set isn't affected; matters most for
task 3.1); coverage gaps (successful email reset, staging rollback, an ops decline case, logs-then-decide).

### Practice baseline (the "Done when" check): passed

```bash
# run_eval now saves git_diff.patch in the run folder when the tree is dirty (these runs: commit 699aaeb + patch)
for setup in base_raw base_seatbelt lightning; do for api in helpdesk ops; do
  .venv/bin/python -m eval.run_eval --setup $setup --api $api --split practice --run-id baseline_${setup}_${api}; done; done
.venv/bin/python -m eval.report --run baseline_base_raw_helpdesk ... --output runs/practice_baseline.md
# 36/36 attempts each. base_raw 4/12 helpdesk, 8/12 ops; base_seatbelt 4/12, 8/12; lightning 9/12, 7/12.
# Against the thinking-off test runs: pr-ops-11 and pr-ops-12 went 3/3 -> 0/3 for every setup (the evaluator fixes
# now fail the wrong behaviour); base_raw helpdesk lost 4 attempts with no relevant change (sampling noise).
# Self-check: 3 rescues (two read-only answers given without looking, one reset not yet done), no correct
# question turned into a wrong action. The developer reviewed the cases; base < 11/12 on both APIs. Task 1.2 done.
```

## 2026-10-17: task 2.1 Tasks and Lightning rollouts

Decisions (with the user): each task's origin goes in the writer's `tasks_<name>.config.json`, not in `Case`;
rollouts the self-check sent back to work are rejected, not edited; Claude runs the writer's Lightning calls,
the developer runs collection; the pilot lives in `data/generated/helpdesk/pilot/`, separate from round 1.

### Research: chat formats and Gym's offline-training tutorial

Read Gym's "Offline Training with Rollouts (SFT/DPO)" tutorial (docs.nvidia.com/nemo/gym/latest; experimental,
and its filter needs a `success` field Gym never writes, GitHub issue #2235): collect, filter by reward, and
convert; low temperature and one rollout per task for SFT. Our filter uses `reward` and the trace instead.
Checked the student's chat template (HF cache, pinned revision) and vLLM 0.27.1's `chat_utils.py` and
`FunctionDefinition`: the template needs tool-call arguments as objects (vLLM parses them; `_postprocess_messages`),
renders `reasoning_content` even with thinking off (so the filter drops it), renders each assistant turn as
`<think></think>...`, which matches vLLM's generation prompt with thinking off; vLLM keeps `strict: false` in the
tool definitions, so the filter renders the rollout's own tools. Importing `nemo_gym` sets `HF_HOME` to its own
cache folder when unset, which hid the student's tokenizer; `factory/filter.py` sets the usual one first.

```bash
.venv/bin/python -m factory.sampler --api helpdesk --n 14            # print sampled briefs and gold calls
.venv/bin/python -m factory.sampler --api ops --n 14
# 400 sampled tasks per API: every one passes sampler.task_problems (gold calls run, kinds consistent)
```

### Writer trials (Lightning on :8000, live; scratch outputs in $SCRATCH)

```bash
.venv/bin/python -m factory.writer --api helpdesk --n 10 --name smoke --seed 11 --out-dir $SCRATCH/w1
# kept 9/10 in 4 s; one written task cited another user's ticket (HD-1040)
PYTHONPATH=. .venv/bin/python $SCRATCH/try_write.py helpdesk off 8    # written tasks, thinking off
PYTHONPATH=. .venv/bin/python $SCRATCH/try_write.py helpdesk on 8     # thinking on
# Off: 4/8 had facts that contradict the fixture or gold values the request never states. On: facts held;
# still unstated values (status, priority). Added writer.unstated() checks; writing whole tasks uses thinking on.
PYTHONPATH=. .venv/bin/python $SCRATCH/try_write.py helpdesk on 8     # 8/8 kept
PYTHONPATH=. .venv/bin/python $SCRATCH/try_write.py ops on 8          # 4/8; 4 dropped because "scale" counted as
                                                                      # a tool name; fixed (only names with "_")
```

### Decontamination calibration

```bash
PYTHONPATH=. .venv/bin/python $SCRATCH/calibrate.py 80     # practice vs practice up to 0.629 (pr-ops-05/06),
                                                           # hand paraphrases 0.565-0.956, phrased tasks max 0.704
PYTHONPATH=. .venv/bin/python $SCRATCH/calibrate2.py       # 150 phrased per API: drops helpdesk 12, ops 20, mostly
                                                           # "changes" (rollbacks in incident fixtures = practice answers)
LLM_MODE=record PYTHONPATH=. .venv/bin/python $SCRATCH/calibrate3.py   # same-user scores; rule limited to no-change kinds
```
Settings: text 0.7 against any practice request; same user, fixture and kind 0.25 (no-change kinds only);
same changes (free text and change ticket aside) as a practice case or part of one.

### Build and test

```bash
.venv/bin/python -m pytest -q          # 182 passed (53 new: sampler, writer, decontaminate, collect, filter)
```
The filter's pieces on the task 1.2 rollouts (thinking on, real Lightning output): helpdesk 22/36 kept (8 gave_up,
4 self_check_nudge, 2 answer_mismatch), ops 21/36 (7 gave_up, 3 state, 3 answer, 2 failed_call); every kept one
renders. 82 of Lightning's 199 answers wrote IDs with a non-breaking hyphen (U+2011); `text` now uses ASCII
look-alikes for hyphens and spaces in the assistant's text.

### Pilot: 50 helpdesk tasks

```bash
LLM_MODE=record .venv/bin/python -m factory.writer --api helpdesk --n 50 --name pilot \
  --out-dir data/generated/helpdesk/pilot                   # kept 50/50 (40 sampled, 10 written) in 53 s
.venv/bin/python -m eval.decontaminate --tasks data/generated/helpdesk/pilot/tasks_pilot.jsonl --against practice
# first run at same_user 0.35 kept 45/50 and missed hd-pilot-0016 (E1004, pr-hd-10's situation, score 0.32);
# restored the task file and reran at 0.25: kept 44/50 (4 changes, 2 same_user)
```

```bash
# (developer) Lightning rollouts on the 44 pilot tasks, from commit 57949f4
.venv/bin/python -m factory.collect --tasks data/generated/helpdesk/pilot/tasks_pilot.jsonl --run-id pilot_helpdesk
# 44/44 rollouts in 12 s (gym_eval.log); the tracebacks in gym_env.log are the usual SIGINT shutdown
.venv/bin/python -m factory.filter --run pilot_helpdesk --round 1 --output data/generated/helpdesk/pilot/round_1.jsonl --show 5
# 40 kept of 44 (sampled 31/35, written 9/10); of the 50 slots: 6 decontaminated, 2 state_mismatch, 2 self_check_nudge.
# Rejections: hd-pilot-0017 (written) never states the delivery method ("can't access their email" passed the
# writer's check), so Lightning asked; hd-pilot-0039 Lightning took "screen flickering" to mean the monitor instead
# of asking which device; hd-pilot-0002 and 0037 asked correctly, but the self-check first said a tool call was
# needed (the nudge is in the conversation, so they're rejected).
```

The developer read 10 kept examples (5 from `--show 5`, 5 more covering missing_info and no_tool): they look right,
except hd-pilot-0038 (no_tool): asked for a colleague's mobile number, Lightning offered to look it up given her
employee ID. Nothing changed, so the verifier passed it. Removed the `colleague_private_info` template (the
right answer is a refusal, which the state match can't check). Tightened the delivery check so that naming email
as the problem ("can't access their email", hd-pilot-0017) no longer counts as asking for email delivery.

```bash
.venv/bin/python -m pytest -q          # 193 passed
```
### Round 1 tasks (run both writers at once, in two terminals, from a clean commit)

```bash
LLM_MODE=record .venv/bin/python -m factory.writer --api helpdesk --n 1300 --name round_1
LLM_MODE=record .venv/bin/python -m factory.writer --api ops --n 1500 --name round_1
# Progress from runs/cache/ at 21 min: helpdesk 135/260 written tasks started, ops 115/300; written-task calls
# average ~5,700 completion tokens, 22% retries; Lightning ~1,450 tokens/s at 32 running requests, KV cache ~12%.
```
Documented round 1 (what it contains, files, methodology) in `data/generated/README.md`, with pointers from
CLAUDE.md "Data formats", PROJECT_OVERVIEW.md and PLAN.md task 2.1.

```bash
# helpdesk: kept 1289/1300 in 2664 s (11 written dropped); ops: kept 1472/1500 in 2864 s (24 written, 4 sampled dropped)
# Of the written-task calls, 63/355 ops and 16/386 helpdesk stopped at max_tokens_write (12,288) still thinking
# (finish_reason length, empty content): most "the answer isn't a JSON object" drops.
# Both configs record commit 365ae42, dirty, with an empty git_diff.patch: the writer read git at the end. The runs
# started at 8e036d4; `git diff --stat 8e036d4 365ae42` touches only docs. Fixed: the writer reads git at the start.
.venv/bin/python -m pytest -q tests/test_factory_writer.py      # 15 passed
.venv/bin/python -m eval.decontaminate --tasks data/generated/helpdesk/tasks_round_1.jsonl --against practice
# kept 1198/1289: 75 changes (58 sampled, 17 written), 16 same_user, 0 text (0.99 quantile 0.558)
.venv/bin/python -m eval.decontaminate --tasks data/generated/ops/tasks_round_1.jsonl --against practice
# kept 1149/1472: 295 changes (173 sampled, 122 written), 24 text, 4 same_user; most hit pr-ops-05 (86),
# pr-ops-03 (59), pr-ops-06 (51), pr-ops-02 (47), pr-ops-01 (39), pr-ops-04 (36)
```

```bash
# (human) Lightning rollouts, from commit 49a5232 (clean)
.venv/bin/python -m factory.collect --tasks data/generated/helpdesk/tasks_round_1.jsonl --run-id round1_helpdesk
# Collected 1198/1198; reward 1 on 1101/1198
.venv/bin/python -m factory.collect --tasks data/generated/ops/tasks_round_1.jsonl --run-id round1_ops
# Collected 1149/1149; reward 1 on 1096/1149
.venv/bin/python -m factory.filter --run round1_helpdesk --round 1 --output data/generated/helpdesk/round_1.jsonl
# kept 1062/1198 (sampled 876, written 186). Rejected: 64 state_mismatch, 31 self_check_nudge, 29 answer_mismatch,
# 8 failed_call, 4 gave_up
.venv/bin/python -m factory.filter --run round1_ops --round 1 --output data/generated/ops/round_1.jsonl
# kept 1016/1149 (sampled 871, written 145). Rejected: 48 failed_call (46 unknown_deployment), 33 state_mismatch,
# 32 self_check_nudge, 17 gave_up, 3 answer_mismatch
```
23 of helpdesk's 29 answer_mismatch rejections pass once look-alike hyphens and spaces (U+2010-2012, U+00A0,
U+202F) in Lightning's answer are made ASCII: right answers the verifier's substring check missed. In the six
`runs/baseline_*` practice runs this explains 1 attempt (Lightning, pr-ops-08). Verifier change left open for the
human (it changes scoring for every setup). Task 2.1 done: the human read 10 pilot examples; round 1 has ~1,000 per API.

### Verifier: the answer check reads look-alike hyphens and spaces as ASCII (user's decision)

`gym_env/verify.py`: new `ASCII_LOOKALIKES` (U+2010, U+2011, U+2012 -> "-"; U+00A0, U+202F -> " "; en and em dashes
left alone) and `answer_check()`, used by `check()`; `factory/filter.py` imports the same mapping for `text`. Every
setup is scored the same way; Gym's itops server picks it up at its next start (it runs the repo's gym_env).

```bash
.venv/bin/python -m pytest -q          # 194 passed (new: test_answer_check_reads_look_alike_hyphens_and_spaces_as_ascii)
.venv/bin/python scripts/checks/rescore_answers.py runs/baseline_* runs/round1_helpdesk runs/round1_ops runs/pilot_helpdesk
# Same state match and seatbelt outcome, answer checked again. Only changes: baseline_lightning_ops attempts 21 -> 22
# of 36 (pr-ops-08 #2), cases 7/12 -> 8/12; round1_helpdesk reward-1 rollouts 1101 -> 1124 of 1198. The student's
# baselines, round1_ops and the pilot don't change. PLAN.md's baseline table marks Lightning ops 8/12 with a note.
```

### Re-filter round 1 with the fixed answer check (user's decision)

`factory/filter.py` runs `gym_env.verify.answer_check` again on each stored final answer (new
`gym_env.tasks.rollout_answer`, the same extraction as the server's `final_answer()`), keeping the recorded state
match; the `other` reason is gone (reward 1 is now exactly: no gave_up, state match, answer check).

```bash
.venv/bin/python -m pytest -q          # 196 passed
# old files kept in $SCRATCH/round1_before/, then removed so the filter could write them again
.venv/bin/python -m factory.filter --run round1_helpdesk --round 1 --output data/generated/helpdesk/round_1.jsonl
# kept 1085/1198 (was 1062): +23 written read_only (26 -> 49 of 55); answer_mismatch 29 -> 6. The 1062 earlier
# examples are byte-identical.
.venv/bin/python -m factory.filter --run round1_ops --round 1 --output data/generated/ops/round_1.jsonl
# kept 1016/1149, byte-identical to before
```

Session wrap-up: CLAUDE.md Commands now show the round 1 factory commands and `scripts/checks/rescore_answers.py`;
PLAN.md task 2.1 notes add the `HF_HOME` quirk and what `text` already contains, for task 2.2.

## 2026-10-17: task 2.2 Training and the round 1 evaluation

### Research: TRL 1.14.1, NemotronH in transformers 5.18.0, the student's chat template

```bash
# How SFTTrainer takes pre-tokenized data, builds labels, and computes the loss
grep -n "assistant_masks\|completion_mask\|skip_prepare_dataset\|loss_type" .venv/lib/python3.13/site-packages/trl/trainer/sft_trainer.py
# Result: a dataset with input_ids and labels is used as is (TRL only truncates to max_length and drops fully masked
# rows). The default loss is chunked_nll: the lm_head runs only on tokens with a label, so the 131k vocabulary
# costs memory only on the ~10% of tokens trained. It needs base_model + lm_head, which NemotronH has.
# assistant_only_loss would need {% generation %} markers, which the student's template lacks, so the mask is ours.
grep -n "truncate_history_thinking\|<think></think>\|add_generation_prompt" \
  ~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f/chat_template.jinja
# Result: thinking off, the generation prompt is "<|im_start|>assistant\n<think></think>", and every assistant turn
# after the last user message renders the same way, so each turn's loss starts after </think> and ends with <|im_end|>
# (the tokenizer's eos). warmup_ratio is gone in transformers 5.18.0: warmup_steps below 1 is a fraction.
# Round 1 token counts with the student tokenizer (scratch script): helpdesk median 1,888, max 3,696, 223k of 2.0M
# tokens trained; ops median 1,753, max 3,584, 189k of 1.8M. Every assistant turn starts with the generation prompt
# and no token crosses a turn's edge.
```

### Build and test

`factory/train_lora.py` (settings under `train:` in `configs/factory.yaml`), `gym_configs/r1.yaml` (base_seatbelt with
the model left unset; `eval/run_eval.py` passes `++policy_model_name=<api>_r1`), tests in `tests/test_train_lora.py`
and `tests/test_run_eval.py`.

```bash
.venv/bin/python -m pytest -q          # 206 passed (10 new)
```

### Dry runs next to both vLLM servers (user approved; adapters in $SCRATCH, runs kept in runs/dryrun_train_*)

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits   # 105647 MiB used by the two servers
# Batch 4 x 4 (the first setting), 3 steps on a seeded 80 examples: 9.4 s/step, peak 23,032 MiB reserved
.venv/bin/python -m factory.train_lora --api helpdesk --round 1 --max-steps 3 --limit 80 \
  --run-id dryrun_train_helpdesk --output $SCRATCH/dryrun_helpdesk_adapter
# Worst case: one step on the 20 longest helpdesk examples (up to 3,696 tokens), written to $SCRATCH/longest_helpdesk.jsonl.
# Peak 36,002 MiB reserved; the whole GPU peaked at 142,418 of 143,771 MiB (sampled with nvidia-smi -lms 500). Too close.
.venv/bin/python -m factory.train_lora --api helpdesk --round 1 --data $SCRATCH/longest_helpdesk.jsonl --max-steps 1 \
  --run-id dryrun_train_helpdesk_longest --output $SCRATCH/dryrun_longest_adapter
# Batch 2 x 8 (same 16 per optimizer step; config copy in $SCRATCH/factory_b2.yaml): worst case 24,306 MiB reserved,
# GPU peak 130,724 MiB (13 GB left); random 80: 7.9 s/step, 15,708 MiB, train loss 0.4492 (4 x 4: 0.4493). Adopted.
.venv/bin/python -m factory.train_lora --api helpdesk --round 1 --config $SCRATCH/factory_b2.yaml \
  --data $SCRATCH/longest_helpdesk.jsonl --max-steps 1 --run-id dryrun_train_helpdesk_b2_longest --output $SCRATCH/dryrun_b2_longest_adapter
.venv/bin/python -m factory.train_lora --api helpdesk --round 1 --config $SCRATCH/factory_b2.yaml \
  --limit 80 --max-steps 3 --run-id dryrun_train_helpdesk_b2_random --output $SCRATCH/dryrun_b2_random_adapter
# Full runs at ~8 s/step: helpdesk 130 optimizer steps (1,031 examples, 54 held out), ops 122 (965, 51 held out).
```

### vLLM loads an adapter from train_lora.py (runtime loading on the running student server, then unloaded)

```bash
# Task 0.2 loaded attention+MLP and Mamba in_proj adapters separately; this one has all seven targets.
curl -s -X POST localhost:8001/v1/load_lora_adapter -H 'Content-Type: application/json' \
  -d '{"lora_name": "dryrun_check", "lora_path": "'$SCRATCH'/dryrun_b2_random_adapter"}'   # 200, added
.venv/bin/python scripts/smoke_toolcall.py --model dryrun_check --min-parsed 19
# dryrun_check (live): 20/20 parsed tool calls, pass mark 19: PASS (a 3-step adapter, so this checks loading and
# routing, not training)
curl -s -X POST localhost:8001/v1/unload_lora_adapter -H 'Content-Type: application/json' -d '{"lora_name": "dryrun_check"}'
# 200; /v1/models lists only "student" again
```

### Round 1 training (the human ran it, in their terminal, both vLLM servers up)

```bash
uv run --group train python -m factory.train_lora --api helpdesk --round 1 && \
uv run --group train python -m factory.train_lora --api ops --round 1
# Both ran from commit aa4ba65 with this task's changes uncommitted (git_diff.patch in each run folder).
# helpdesk_r1: 1,031 examples (54 held out), 130 steps in 15.0 min; eval loss 0.1556 -> 0.1473 (epochs 1 -> 2),
#   token accuracy 0.946 -> 0.950; peak 27,582 MiB reserved.
# ops_r1: 965 examples (51 held out), 122 steps in 13.7 min; eval loss 0.2712 -> 0.2608, accuracy 0.910 -> 0.913;
#   peak 26,718 MiB reserved. Train loss noisy but falling in both; gradient norm under 0.4 throughout.
# Adapters in adapters/helpdesk_r1 and adapters/ops_r1 (68 MB each); configs/models.yaml lists them.
```

### Restart with the adapters, smoke checks, r1 evals (the human ran the chain; the evals failed to start)

```bash
scripts/stop_vllm.sh student && scripts/serve_student.sh && \
uv run python scripts/smoke_toolcall.py --model helpdesk_r1 --min-parsed 19 && \
uv run python scripts/smoke_toolcall.py --model ops_r1 --min-parsed 19 && \
uv run python -m eval.run_eval --setup r1 --api helpdesk --split practice --run-id r1_helpdesk && \
uv run python -m eval.run_eval --setup r1 --api ops --split practice --run-id r1_ops
# Student ready after 34 s serving helpdesk_r1 and ops_r1. Smoke: helpdesk_r1 20/20, ops_r1 20/20 (pass mark 19).
# r1_helpdesk: "Gym servers didn't start" (Process `itops` finished unexpectedly); the chain stopped, so r1_ops didn't run.
# Cause: Ray's uv hook. Launched under `uv run`, each Gym server's ray.init() rebuilt the uv project for Ray's workers
# and raised "pyproject.toml is not in the working_dir .../resources_servers/itops". Every earlier eval and collection
# ran as .venv/bin/python, which never triggers it; CLAUDE.md's Commands showed the `uv run` form.
```

### Fix: eval/run_eval.py starts Gym with Ray's uv hook off

```bash
# Reproduced without servers: the step of ray.init() that applies the hook, run from resources_servers/itops
uv run python $SCRATCH/ray_uv_hook_check.py                                  # RuntimeError: Your .../pyproject.toml is not in the working_dir ...
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 uv run python $SCRATCH/ray_uv_hook_check.py  # ok, runtime_env {}
../../.venv/bin/python $SCRATCH/ray_uv_hook_check.py                         # ok, runtime_env {} (how the baselines ran)
# start_servers() now passes RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 (Gym copies os.environ into every server), so the
# servers start as in the baselines however run_eval is launched. gym eval run --no-serve doesn't start Ray.
.venv/bin/python -m pytest -q          # 207 passed
# The new start_servers test against the committed code (65efc94), without the fix: 1 failed. Then the fix back.
git stash push eval/run_eval.py && .venv/bin/python -m pytest -q tests/test_run_eval.py -k uv_hook; git stash pop
mv runs/r1_helpdesk runs/r1_helpdesk_failed_uv_hook   # the failed run's folder, kept; frees the run id
```

### r1 evals (the human ran them after the fix) and the ladder table: the Done-when check passed

```bash
.venv/bin/python -m eval.run_eval --setup r1 --api helpdesk --split practice --run-id r1_helpdesk && \
.venv/bin/python -m eval.run_eval --setup r1 --api ops --split practice --run-id r1_ops
# r1 on helpdesk practice: 7/12 cases passed; mean latency 5.5 s; $0.17 per 1,000 requests (36/36 attempts)
# r1 on ops practice: 8/12 cases passed; mean latency 7.3 s; $0.28 per 1,000 requests (36/36 attempts)
# Same eval_config as the baselines (compared config.json). Against base_seatbelt: helpdesk gains pr-hd-03, 06, 07,
# loses none (attempts 24/36 vs 16/36); ops gains pr-ops-02, loses pr-ops-06 (to_revision 7 instead of 8 in 2 of 3),
# attempts 25/36 vs 23/36. Completion tokens and calls per attempt about the same; time per completion token about a
# third higher (helpdesk 22.8 vs 17.1 ms, ops 20.2 vs 16.3), likely LoRA serving overhead.
.venv/bin/python -m eval.report --run baseline_base_raw_helpdesk --run baseline_base_seatbelt_helpdesk --run r1_helpdesk \
  --run baseline_lightning_helpdesk --run baseline_base_raw_ops --run baseline_base_seatbelt_ops --run r1_ops \
  --run baseline_lightning_ops --output runs/practice_ladder_r1.md
# PLAN.md's ladder table now has the r1 row. Smoke 20/20 for both adapters (above). Task 2.2's checks pass; the human
# commits and tags fallback-v1.
# The per-case, per-attempt and latency comparisons above came from short read-only Python over runs/<run>/config.json,
# per_case.json, traces.jsonl and rollouts.jsonl (baseline_*_<api> against r1_<api>), not kept.
```

## 2026-10-17: task 3.1 Specialists and the Lightning orchestrator

### Research and checks

```bash
# Which models the running vLLM servers serve (team_trained needs both adapters)
curl -s -m 3 http://localhost:8000/v1/models   # ['lightning']
curl -s -m 3 http://localhost:8001/v1/models   # ['student', 'helpdesk_r1', 'ops_r1']
# Gym 0.6.0's remote_agent (README and app.py): a service at /v1/responses returns each tool call to Gym, which runs it
# and calls the service again; the trace and the give-up rule would sit outside Gym's verify response. Decided (the
# user): the team is its own Gym agent, itops_team. Gym's vllm_model server always sends its configured model name
# (app.py: body.model = self.config.model), so the planner and each specialist get their own model server.
# The plan schema (enum of active specialists, maxItems 6): vLLM's xgrammar feature check passes and it compiles
.venv-vllm/bin/python -I -c "...has_xgrammar_unsupported_json_features(schema); xgr.Grammar.from_json_schema(...)"   # in $SCRATCH
```

### Ticket fixture

```bash
# configs/fixtures.yaml: ticket_fixtures (incident_prod_checkout plus 5 hand-written tickets). Generated into $SCRATCH
# first: every existing fixture and logs.json keep identical state; only generated_by.config gains ticket_fixtures.
.venv/bin/python -m gym_env.fixtures.make_fixtures --config configs/fixtures.yaml --out-dir $SCRATCH/fixtures_check
.venv/bin/python -m gym_env.fixtures.make_fixtures --config configs/fixtures.yaml
# New: incident_prod_checkout_tickets, HD-1042..HD-1046, next ticket HD-1047. Regenerated once more after dropping
# "1.20.0" from the checkout ticket's summary, so pr-multi-06 needs ops to find the image.
```

### Build and test

Cards and registry (`agents/cards/`, `agents/registry.json`), `agents/specialist.py`, `agents/orchestrator.py` (plan,
run order, results between steps, CLI), `ItopsTeamAgent` in `common/gym_agent.py` (the single agent's model, tool,
seed and verify code moved into shared helpers, unchanged), `responses_api_agents/itops_team/`,
`gym_configs/eval_team.yaml`, `team_base.yaml`, `team_trained.yaml`, team runs in `eval/run_eval.py`, `Step` tokens
and `Trace.plan` in `common/schemas.py`, and 8 cases in `data/practice/multi.jsonl` (drafted by Claude, for the
human's review). Tests: `tests/test_orchestrator.py`, `tests/test_gym_agents.py` (both Gym agents' /run in process
against the real resources server, scripted model servers), additions to `tests/test_run_eval.py` and
`tests/test_gym_env.py`.

```bash
# Every setup's Gym config resolved offline, as `gym env start` resolves it (no servers started; $SCRATCH script on
# GlobalConfigDictParser with offline=True). Team setups failed: itops.yaml's smoke agent points at policy_model,
# which only eval.yaml defines. eval_team.yaml now points it at planner_model; all 6 setups resolve. Kept as
# test_gym_resolves_every_setup_s_config, which fails on the old eval_team.yaml (checked) and passes now.
.venv/bin/python $SCRATCH/gym_parse_check.py base_raw base_seatbelt lightning r1 team_base team_trained
.venv/bin/python -m pytest -q          # 251 passed (44 new)
```

### The demo request through the team CLI (approved by the user: once, on the running servers)

```bash
LLM_MODE=record .venv/bin/python -m agents.orchestrator --case pr-multi-01 --setup team_trained
# PASS in 3.4 s. Plan (Lightning, 1,245 + 237 tokens): 1. helpdesk opens a P1 incident ticket, 2. ops (after 1) rolls
# payments back with step 1's ticket. helpdesk_r1: create_ticket -> HD-1042. ops_r1: list_deployments,
# rollout_history, rollback to 8 with change_ticket HD-1042. Self-checks complete; no seatbelt retries.
LLM_MODE=replay .venv/bin/python -m agents.orchestrator --case pr-multi-01 --setup team_trained
# Same plan, calls and PASS from runs/cache/, with no server; prod/payments has 4 Running pods after the run.
```

### Team evals on the multi-API practice cases (the human ran them, commit 0aa4e7b, clean tree)

```bash
uv run python -m eval.run_eval --setup team_base --split practice --run-id team_base_practice && \
uv run python -m eval.run_eval --setup team_trained --split practice --run-id team_trained_practice
# team_base: 4/8 cases (12/24 attempts), mean latency 19.1 s, $0.86 per 1,000 requests; failed attempts 5 routing, 7 specialist
# team_trained: 2/8 cases (11/24 attempts), 18.3 s, $0.75; failed attempts 4 routing, 9 specialist. 24/24 attempts each,
# no Gym failures, under a minute each. Case by case (base / trained): 01 0/3 3/3, 02 2/3 1/3, 03 1/3 1/3, 04 3/3 0/3,
# 05 2/3 1/3, 06 3/3 3/3, 07 0/3 1/3, 08 1/3 1/3.
.venv/bin/python -m eval.report --run team_base_practice --run team_trained_practice
# Per attempt (plan, instructions, calls, seatbelt events, state diff, final answer): read-only scripts in $SCRATCH
# over runs/team_*_practice/rollouts.jsonl and traces.jsonl.
.venv/bin/python $SCRATCH/team_attempts.py team_base_practice team_trained_practice
.venv/bin/python $SCRATCH/team_detail.py team_trained_practice pr-multi-02 1 pr-multi-02 2 pr-multi-03 1 pr-multi-07 1 pr-multi-08 1
# Findings: every routing failure is pr-multi-07 or 08, where Lightning planned the work (opening tickets nobody asked
# for, scaling prod to 0, rolling back to "revision 1") instead of asking or declining; the planner prompt's example
# ("found or opened in an earlier step") invites the extra tickets. Several "specialist" failures are the planner's
# instructions too: no acting user (helpdesk_r1 asked for an employee ID, pr-multi-02 #1), "category software" for a
# requested incident ticket (pr-multi-03 #1), all categories listed (pr-multi-02 #2), the card's tags copied. The labels
# (CLAUDE.md "Scoring") only compare the plan's agents with the gold APIs. Specialists: the student base attaches the
# user's laptop to incident tickets (pr-multi-01 0/3, as in 1.2's baseline); the adapters don't (3/3) but act less
# often on the planner's instructions: steps with no tool call 16 of 74 (team_trained) against 5 of 71 (team_base).
```

### Planner prompt revision (2026-10-18; the user's decision after the first practice runs)

`PLANNER_PROMPT` in `agents/orchestrator.py`, for the failures above; the plan schema is unchanged. `analysis` now
starts with the actions the user asked for and whether to carry them out, ask or decline. New rules: only the actions
asked for (a ticket only when asked; no checks, follow-ups or "if" steps); a prod change needs a ticket the user gave,
pointed to or asked to open, otherwise ask for one; decline what would cause an outage in prod or undo work that isn't
the user's, and ask only for what no specialist can look up; one step per action; instructions written as the user's
request (the employee named, the user's details kept exactly, no questions, no card fields, values from earlier
steps taken from their results). The demo responses recorded on 2026-10-17 in runs/cache/ no longer match the
planner's request, so replay needs a new recording (task 5.3).

```bash
.venv/bin/python -m pytest -q          # 251 passed
curl -s -m 3 http://localhost:8000/v1/models   # ['lightning']; 8001: ['student', 'helpdesk_r1', 'ops_r1']; port 11000 free
```

### Team evals after the planner revision (the human ran them, commit 60bc6b5; dirty only for .claude/settings.json, deleted in the working tree)

```bash
uv run python -m eval.run_eval --setup team_base --split practice --run-id team_base_practice_v2 && \
uv run python -m eval.run_eval --setup team_trained --split practice --run-id team_trained_practice_v2
# team_base: 5/8 cases (14/24 attempts), 10.2 s, $0.60 per 1,000 requests; failed attempts 2 routing, 8 specialist
# team_trained: 6/8 cases (17/24 attempts), 16.7 s, $0.59; failed attempts 1 routing, 6 specialist. 24/24 each, no Gym
# failures. Cases (base / trained): 01 1/3 3/3, 02 2/3 1/3, 03 1/3 2/3, 04 1/3 2/3, 05 2/3 1/3, 06 2/3 3/3, 07 2/3 2/3, 08 3/3 3/3.
.venv/bin/python -m eval.report --run team_base_practice --run team_trained_practice --run team_base_practice_v2 --run team_trained_practice_v2
.venv/bin/python $SCRATCH/team_attempts.py team_base_practice_v2 team_trained_practice_v2   # and team_detail.py on the failures
# v1 -> v2: mean plan steps 3.2/3.3 -> 2.3; attempts with a step that gave up 4/5 -> 0/1; team_trained steps without a
# tool call 16 of 74 -> 5 of 56; attempts opening an unrequested ticket (pr-multi-04..08) base 4/15 -> 3/15 (all asked for
# by Lightning's plan), trained 4/15 -> 0/15. pr-multi-08's 3/3 comes from nothing changing, not from declining: one
# step-free plan (its reply repeats the request), three look-up-only plans, and two that planned the scale to 0, where
# ops invented change tickets (HD-1234, HD-001) that the environment refused; E1012 has no open tickets to close.
# Remaining: Lightning turns a requested incident ticket into hardware or software (team_trained 02 #1, #2, 03 #3),
# leaves the change ticket out of a rollback step (04 #3), asks ops only to inspect in pr-multi-05; the seatbelt
# self-check failed a step that rightly made no call (07 #1, state correct, scored 0).
```

### Restore the test-set deny rules (the user's decision)

```bash
# .claude/ was missing from the working tree (only .claude/settings.json is tracked; git status was clean at the start
# of the session, so nothing untracked was in it). Restored from HEAD; identical to the committed file.
git checkout -- .claude/settings.json
git diff --quiet HEAD -- .claude/settings.json && echo "matches HEAD"   # matches HEAD
```

## 2026-10-18: task 3.2 System baselines

### Planning: what re-rendering the training examples with every tool costs (CPU only, read-only)

```bash
# For single_trained's adapter (the user chose this 2026-10-18): train on the same trajectories, rendered with every
# API's tools as the single_* setups see them at eval. $SCRATCH/tool_tokens.py re-renders round 1's examples with
# factory.filter.render, with each example's own API's tools and then with every tool.
PYTHONPATH=$PWD .venv/bin/python $SCRATCH/tool_tokens.py
# Own API's tools: identical to the stored text for helpdesk 1,085/1,085 and ops 1,016/1,016 (gym_env.env's
# Responses-format schemas render exactly like the chat-format tools filter.py had). Every tool: helpdesk tokens
# 2,025,176 -> 3,212,166 (1.59x), ops 1,801,570 -> 3,024,834 (1.68x); longest 3,696 -> 4,790 (max_length is 8,192).
# Data unchanged since the specialists trained: sha256 of both round_1.jsonl files matches runs/train_*_r1/config.json.
```

### single_base and single_lightning

```bash
# gym_configs/single_base.yaml and single_lightning.yaml are base_seatbelt.yaml's and lightning.yaml's contents;
# eval/run_eval.py already runs single_* setups on the multi-API cases with itops_agent, and each case row carries
# all 12 tools (helpdesk and ops). Tests: those files equal the ladder's, the rows carry every tool, single_* refuse
# --api, and the parametrized offline Gym config check now covers both setups.
.venv/bin/python -m pytest -q          # 255 passed (4 new)
```

### single_trained's adapter: `factory/train_lora.py --api all` (re-rendered with every tool, the user's choice)

```bash
# --api all: both APIs' round_<n> files; each API's examples split on their own with the seed (so the held-out
# examples are the round 1 adapters'); each trajectory rendered again with every tool after checking that its own
# API's tools give back its stored text. Adapter all_r<n>, run train_all_r<n>; config.json and training_run.json say
# how the text was rendered. Tests: the one-API split is unchanged and the union's is both APIs' splits, --api all
# needs every API's examples, and (real template and tokenizer) the new text lists every tool and trains the same tokens.
.venv/bin/python -m pytest -q          # 258 passed (3 new)
# The full round 1 data through the same functions ($SCRATCH/union_check.py, CPU only, 40 s): 1,996 training examples
# and 105 held out, the held-out ids identical to runs/train_{helpdesk,ops}_r1's; trained tokens 392,801 (train) and
# 19,285 (held out), the round 1 adapters' sums, and unchanged in each of the 2,101 examples. Training tokens
# 3,638,423 -> 5,928,197 (1.63x), longest 4,790. Also wrote the 10 longest per API to $SCRATCH/longest_all.jsonl.
PYTHONPATH=$PWD .venv/bin/python $SCRATCH/union_check.py $SCRATCH/longest_all.jsonl
```

### Dry run next to both vLLM servers (the user approved; adapter in $SCRATCH, run kept in runs/dryrun_train_all_longest)

```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader   # 104,987 MiB (Lightning 81,616, student 23,356), 0 %
ss -ltn | grep -c ':11000 '                                                 # 0: no eval running
# Worst case: one step on those 20 examples (16 per step, 3,784-4,790 tokens); the GPU sampled every 500 ms with
# nvidia-smi --query-gpu=memory.used -lms 500 into $SCRATCH/dryrun_all_gpu.txt, stopped after the run.
.venv/bin/python -m factory.train_lora --api all --round 1 --data $SCRATCH/longest_all.jsonl --max-steps 1 \
  --run-id dryrun_train_all_longest --output $SCRATCH/dryrun_all_longest_adapter
# 15.2 s for the step (2.2's helpdesk worst case: 11.7 s); peak 18,746 MiB allocated and 24,406 reserved (2.2: 16,377
# and 24,306), 37,644 free at the start; the whole GPU peaked at 130,164 of 143,771 MiB (2.2: 130,724), and went back
# to 104,987. The round 1 runs reserved about 3 GB more than their dry run, so expect about 28 GB. Time: 1.63x the
# round 1 runs' tokens at their 6.8 s per step, about 11 s x 250 steps: about 47 min.
```

### single_trained

```bash
# gym_configs/single_trained.yaml: single_base.yaml with model all_r1 (on the student server); configs/models.yaml
# records the adapter. Tests: it differs from single_base only in the model, all_r1 is priced as the student, and the
# offline Gym config check covers it.
.venv/bin/python -m pytest -q          # 259 passed (1 new)
```

### all_r1 training (the human ran it, in their terminal, both vLLM servers up)

```bash
uv run --group train python -m factory.train_lora --api all --round 1
# Ran from commit 744d3ca with this task's changes uncommitted (git_diff.patch in runs/train_all_r1/). 1,996 examples
# re-rendered with every tool, 105 held out (the round 1 adapters' ids, checked in its config.json); 250 steps in
# 43.4 min (10.4 s/step); train loss 0.2741; eval loss 0.2097 -> 0.2021 (epochs 1 -> 2), token accuracy 0.9295 ->
# 0.9324; peak 30,238 MiB reserved, 18,878 allocated, 37,644 free at the start. On the same 105 examples the round 1
# adapters ended at 0.1473 (helpdesk's 54) and 0.2608 (ops's 51), 0.2025 weighted by examples. Adapter in
# adapters/all_r1 (68 MB; r=16, alpha 32, the same seven targets).
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader   # 104,987 MiB, 0 %: training released its memory
curl -s localhost:8001/v1/models   # student, helpdesk_r1, ops_r1: all_r1 is served only after the student server restarts
```

### single_* evals and the all_r1 smoke check (the human ran the chain, commit e17ec08, clean tree)

```bash
uv run python -m eval.run_eval --setup single_base --split practice --run-id single_base_practice && \
uv run python -m eval.run_eval --setup single_lightning --split practice --run-id single_lightning_practice && \
scripts/stop_vllm.sh student && scripts/serve_student.sh && \
uv run python scripts/smoke_toolcall.py --model all_r1 --min-parsed 19 && \
uv run python -m eval.run_eval --setup single_trained --split practice --run-id single_trained_practice
# 24/24 attempts each, every step on the setup's model (student, lightning, all_r1). all_r1 (live): 20/20 parsed,
# pass mark 19: PASS (runs/smoke_toolcall/all_r1_live_20261018-035620). The student server now serves student,
# all_r1, helpdesk_r1 and ops_r1.
# single_base 4/8 cases (13/24 attempts), 7.1 s, $0.51 per 1,000 requests; single_trained 5/8 (16/24), 7.3 s, $0.36;
# single_lightning 7/8 (18/24), 9.5 s, $0.87.
```

### The system practice table: the Done-when check passed

```bash
uv run python -m eval.report --run single_base_practice --run single_trained_practice --run single_lightning_practice \
  --run team_base_practice_v2 --run team_trained_practice_v2 --output runs/practice_system.md
# | setup            | cases passed | attempts | mean latency (s) | $ per 1,000 requests | run                       |
# | single_base      | 4/8          | 24/24    | 7.1              | 0.51                 | single_base_practice      |
# | single_trained   | 5/8          | 24/24    | 7.3              | 0.36                 | single_trained_practice   |
# | single_lightning | 7/8          | 24/24    | 9.5              | 0.87                 | single_lightning_practice |
# | team_base        | 5/8          | 24/24    | 10.2             | 0.60                 | team_base_practice_v2     |
# | team_trained     | 6/8          | 24/24    | 16.7             | 0.59                 | team_trained_practice_v2  |
# Cases (base / trained / lightning / team_base / team_trained): 01 0/3 1/3 2/3 1/3 3/3, 02 0/3 3/3 3/3 2/3 1/3,
# 03 2/3 3/3 3/3 1/3 2/3, 04 1/3 0/3 3/3 1/3 2/3, 05 1/3 1/3 2/3 2/3 1/3, 06 3/3 3/3 3/3 2/3 3/3, 07 3/3 3/3 2/3 2/3 2/3,
# 08 3/3 2/3 0/3 3/3 3/3. Attempts 13, 16, 18, 14, 17 of 24.
# Per failed attempt (calls, seatbelt events, state diff, answer): a read-only script over the runs' rollouts.jsonl.
.venv/bin/python -I $SCRATCH/failures.py single_trained_practice single_base_practice single_lightning_practice
# single_trained: 7 of 8 failures need a helpdesk ticket ID in a prod change (pr-multi-01, 04, 05): asked the user for
# a change ticket 5 times (once after finding HD-1043), invented HD-12345 once, stopped after list_tickets once; the
# 8th began scaling prod to 0 on pr-multi-08 and the seatbelt gave up. single_base: incident tickets as hardware or
# with a device (4), claimed rollbacks or scales never called (3), wrong deployment (1), an extra note (1), out of
# retries or turns (2). single_lightning: pr-multi-08 0/3 (once scaled all of prod to 0 under its own P4 ticket,
# twice a look-up loop to max turns), a look-up loop on 07, a hardware ticket with the laptop on 01, a question on 05.
# The order of create_ticket and rollback on pr-multi-01 and 02 (inline script over the same rollouts): all_r1 follows
# the request's order (02, ticket named first: 3/3 ticket -> rollback; 01, rollback named first: 3/3 rollback first,
# 400 from the change-ticket rule, one recovered); Lightning opened the ticket first in all 6 attempts.
# Where ops's training data gets prod change tickets (inline script): all 482 prod rollback/scale calls in
# data/generated/ops/round_1.jsonl take change_ticket from the user's request; the ops examples call only ops tools.
```

## 2026-10-18: task 3.3 Escalation and Lightning's planner contribution

### team_small_planner (gym_configs/team_small_planner.yaml: team_trained with the student base model as planner)

```bash
curl -s localhost:8001/v1/models; curl -s localhost:8000/v1/models   # 8001: student, all_r1, helpdesk_r1, ops_r1; 8000: lightning
# Handed over for an early run while escalation was built, but not run; it moved into the chain at the final commit.
```

### Escalation checks without a GPU

```bash
# pr-multi-01 through the CLI from the replay cache, before handing over the evals: both miss on the planner's first
# request (the same cache key, f11b138f...), because the recording (task 3.1) predates the planner prompt revision.
# So team_escalate sends exactly team_trained's requests; the live check is in the human's chain below.
LLM_MODE=replay uv run python -m agents.orchestrator --case pr-multi-01 --setup team_trained    # ReplayMiss
LLM_MODE=replay uv run python -m agents.orchestrator --case pr-multi-01 --setup team_escalate   # ReplayMiss
```

### The three team setups at one commit (handed over to the human; both vLLM servers up, port 11000 free)

```bash
# One live demo request on team_escalate (also refreshes the replay cache for pr-multi-01), then the three evals.
LLM_MODE=record uv run python -m agents.orchestrator --case pr-multi-01 --setup team_escalate && \
uv run python -m eval.run_eval --setup team_trained --split practice --run-id team_trained_practice_v3 && \
uv run python -m eval.run_eval --setup team_escalate --split practice --run-id team_escalate_practice && \
uv run python -m eval.run_eval --setup team_small_planner --split practice --run-id team_small_planner_practice
```
# Started 06:59:57 UTC, all three done by about 07:02. 24/24 attempts each, commit 954c591 with this task's changes
# uncommitted (each run's git_diff.patch equals the tree's tracked diff; the two new setup files are untracked and
# predate the runs). The demo request on team_escalate: PASS, no escalation; it now replays with no server.
# team_trained v3 6/8 (16/24), 11.8 s, $0.54; team_escalate 3/8 (11/24), 12.0 s, $0.60, 49/51 small-model steps
# (2 escalated: pr-multi-05 #2 rescued, pr-multi-01 #3 not); team_small_planner 3/8 (11/24), 5.0 s, $0.28.
```

### The system practice table: the Done-when check passed

```bash
.venv/bin/python -m eval.report --run single_base_practice --run single_trained_practice --run single_lightning_practice \
  --run team_base_practice_v2 --run team_trained_practice_v2 --run team_trained_practice_v3 \
  --run team_escalate_practice --run team_small_planner_practice --output runs/practice_system.md
# Per failed or escalated attempt (plan, steps, escalations, state diff): a read-only script over the runs' traces and
# rollouts; failure counts (routing/specialist): team_trained v3 1/7, team_escalate 4/9, team_small_planner 7/6.
.venv/bin/python -I $SCRATCH/team_attempts.py team_escalate_practice team_trained_practice_v3 team_small_planner_practice
# Lightning's escalated parts: $0.041 per 1,000 requests; the specialists' parts at Lightning's rate would add $0.023.
LLM_MODE=replay uv run python -m agents.orchestrator --case pr-multi-01 --setup team_escalate   # PASS, from the cache

### git_diff.patch with untracked files, never data/test/ (the user's request, after the runs)

```bash
uv run pytest -q   # 278 passed, with the new throwaway-repo test of git_commit
# git_commit on the real repo into $SCRATCH/patch_check (deleted after): HEAD 9e34f45, dirty; the patch covered exactly
# the two changed files (eval/run_eval.py, tests/test_run_eval.py); every "data/test" in it is this change's own code
# and comments; `git diff --cached --name-only` was the same before and after.
.venv/bin/python -c "from pathlib import Path; from eval.run_eval import git_commit; print(git_commit(Path('$SCRATCH/patch_check')))"
```
