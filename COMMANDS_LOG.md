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
