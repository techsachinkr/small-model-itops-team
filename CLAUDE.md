# CLAUDE.md

## Project

**Small-model IT operations team**, our entry for the NVIDIA x Red Hat Small Models Hack (Raleigh, October 17–18, 2026). Main track: Multi-Agent Systems. We are also targeting three bounties: Best Model Improvement with NeMo Gym, Best Use of Nemotron Lightning, and Best Use of Red Hat vLLM.

Claim we are testing: Nemotron 3.5 Lightning plans and teaches, and small specialist models, trained with NeMo Gym on Lightning's verified work, make the tool calls. The team beats one small model with every tool, and gets close to Lightning working alone while each specialist needs a fraction of the memory.

Parts:

- **Environment (NeMo Gym)**: one resources server with the `helpdesk` and `ops` tools (`calendar` is optional), isolated state per task, and a verifier that compares the final state with the state produced by the task's gold calls.
- **Factory**: generate tasks → Lightning solves them as NeMo Gym rollouts → keep only rollouts the verifier passes → LoRA fine-tune a small student per API → a failure analyst (Lightning) groups mistakes on practice tasks → targeted tasks in the next round.
- **Team**: a Lightning orchestrator reads specialist cards from a registry, splits a request into steps, passes results between steps (for example a ticket ID), and hands each step to a specialist.
- **Seatbelt**: a reliability layer inside the shared tool loop.
- **Escalation**: when the seatbelt gives up on a step, Lightning finishes it.
- **Serving (vLLM)**: Lightning and every specialist adapter are served by vLLM, with measured serving results.

## Hard rules

1. **All code must be written during the hackathon** (event rule). Outside code is allowed only if it has been open source for 2+ weeks, which covers normal libraries. Never paste in code written before the event.
2. **Every generative model is ≤33B total parameters** (event rule), including any used for verification, critique, routing or speculative decoding. For MoE models, total parameters count, using the official number on the model card. Embedding and reranking models are excluded. The limit is per model, not combined (organizers, 2026-10-17; PLAN.md task 0.1). Use only models listed in `configs/models.yaml`.
3. **Open data** (event rule): every synthetic dataset is saved with its generation config and seed so it can be published with its methodology. Every outside data source is listed in `data/open/DATA_SOURCES.md` with its link and license.
4. **`data/test/` is locked.** Never open, print, summarize, edit, or generate data from anything in it. Only `eval/run_eval.py --split test --final` and `eval/decontaminate.py --against test` may read it, and only when a human asks for that run. If a task seems to need test data, stop and ask.
5. **The factory never sees test data.** The failure analyst reads only results on `data/practice/`.
6. **Correctness is checked by the verifier's state match, never by an LLM judge.** See "Scoring".
7. **Every setup runs under the same conditions**: same NeMo Gym eval config, same tools per API, same seatbelt settings, same max steps, same temperature, same attempts per case. Only the model, adapter or agent changes between setups. Only `team_escalate` escalates. Only `base_raw` runs without the seatbelt.
8. **Only report numbers produced by a script you just ran.** Results come from files in `runs/`. Say clearly when a run failed or was partial.
9. **Work only on the task you were given.** Check `PLAN.md`. Don't start the next task until the current task's "Done when" check passes. `main` must always run.
10. **Ask before adding a framework or dependency** that isn't listed under Stack.

## Stack

- Python 3.13 (NeMo Gym needs ≥3.13.14), `uv` for environment and dependencies. `pyproject.toml` sets `exclude-newer` to 2026-10-03, so every dependency was public at least 2 weeks before the event (rule 1). NeMo Gym is pinned to 0.6.0 and installs only on Linux (it needs uvloop), so Gym, vLLM and training run on the Brev box.
- NeMo Gym: the environment is a Gym resources server, and all evaluation and rollout collection go through Gym. Follow the installed version's docs, especially the Stateful Environment tutorial, the Execution and State Match verification pattern, the Local vLLM model server page, and Offline Training (SFT/DPO). Don't guess Gym APIs.
- vLLM: version 0.27.1 (the version on Lightning's model card), in its own venv `.venv-vllm/` because it pins torch 2.13.0 and the train group uses 2.14.1. Lightning on port 8000. Student base model plus LoRA adapters on port 8001 (`--enable-lora`, `--lora-modules name=path`). Tool calling via `--enable-auto-tool-choice --tool-call-parser <parser>`, with each parser set in `configs/models.yaml`. Use structured outputs (JSON schema) for orchestrator plans, seatbelt self-checks and analyst output. Check the installed version's defaults for prefix caching. Use speculative decoding for Lightning only if the installed vLLM supports it for this model, and any draft model counts toward rule 2.
- Training: Hugging Face TRL + PEFT (LoRA), or Unsloth following Gym's Unsloth tutorial. Loss on assistant turns only.
- Team: plain Python. Specialist cards are JSON files in `agents/cards/` using the same fields as an A2A Agent Card. Wrapping specialists with the A2A SDK is a stretch goal.
- Benchmarks: GuideLLM or vLLM's own benchmark tools.
- GPU: one Brev instance. `configs/gpu.md` records which servers share the GPU and their memory settings. Training scripts may need to pause a server first.

Ports: Lightning 8000, student 8001, orchestrator 8200. Gym's own ports are set in its config.

## Repo layout

```
configs/          models.yaml, eval.yaml, gpu.md
gym_env/          resources server: state.py, tools.py, helpdesk.py, ops.py, calendar.py (optional),
                  env.py (one task's tools + state), server.py (Gym server), verify.py, chaos.py,
                  tasks.py (our schema → Gym format), fixtures/ (default.json, make_fixtures.py)
resources_servers/itops/  Gym's required location: app.py starts gym_env/server.py; config, example data
gym_configs/      NeMo Gym configs, one per setup
common/           llm.py (every model call), seatbelt.py, tool_loop.py, schemas.py
factory/          sampler.py, writer.py, collect.py (Lightning rollouts via Gym), filter.py,
                  train_lora.py, analyst.py, run_round.py
agents/           specialist.py, orchestrator.py, registry.json, cards/
eval/             run_eval.py (wraps Gym), decontaminate.py, report.py
bench/            vLLM serving benchmarks
data/open/        outside open data (Loghub log samples, Workplace Assistant and When2Call subsets),
                  DATA_SOURCES.md
data/practice/    hand-written cases; the analyst may read results on these
data/test/        hand-written cases; LOCKED
data/generated/   synthetic data, to be published
adapters/         LoRA outputs (gitignored)
runs/             traces, scores, reports, cache/ (gitignored except runs/final/)
demo/             app.py
scripts/          serve_lightning.sh, serve_student.sh, stop_vllm.sh, smoke_toolcall.py, demo_reset.sh
tests/            pytest tests
```

## Data formats (defined in `common/schemas.py`)

Our schemas are the source of truth. Conversion to NeMo Gym's input format happens only in `gym_env/tasks.py`.

Case (`data/practice/*.jsonl`, `data/test/*.jsonl`, `data/generated/**/tasks_*.jsonl`):
`{id, request, user_id, fixture, apis, gold_calls: [{api, tool, args}], kind, answer_must_contain?: [str]}`
where `kind` is one of `single | multi | no_tool | missing_info | read_only`.

Training example (`data/generated/<api>/round_<n>.jsonl`):
`{id, api, round, kind, origin: sampled|written, source: broad|targeted, cluster?, case_id, teacher, reward, messages, text}`
`sampled` means code picked the action and computed `gold_calls`, and a model only phrased the request. `written` means a model wrote both. `messages` is Lightning's full trajectory including tool results. `text` is that trajectory rendered with the student base model's chat template.

Trace (`runs/<run_id>/traces.jsonl`):
`{run_id, setup, case_id, attempt, steps: [{agent, model, calls, seatbelt_events, escalated}], prompt_tokens, completion_tokens, latency_s, cost_usd, passed, failure_label?: routing|specialist}`

Run summary (`runs/<run_id>/summary.json`): per setup, passes as `"k/n"`, mean latency, cost per 1,000 requests, and for `team_escalate` the share of steps handled by small models.

Failure cluster (`runs/<run_id>/clusters.json`): `[{name, description, case_ids, guidance}]`

## Eval setups (names used everywhere)

Specialist ladder, per API, on that API's single-API cases. Each row changes one thing, so improvements are isolated:
- `base_raw`: student base model, no seatbelt
- `base_seatbelt`: student base model with the seatbelt
- `r1`: round 1 adapter
- `r2`: round 2 adapter
- `lightning`: Lightning, as a reference

System, on multi-API cases:
- `single_base`, `single_trained`, `single_lightning`: one model with every tool and no orchestrator (`single_trained` is one adapter trained on all specialists' data)
- `team_base`: Lightning orchestrator with untrained specialists
- `team_trained`: Lightning orchestrator with trained specialists
- `team_escalate`: `team_trained` plus escalation
- `team_small_planner` (optional): `team_trained` with the student base model as orchestrator, to measure Lightning's contribution as planner

## Environment rules

- Deterministic: IDs come from per-table counters set by the fixture, and the clock is frozen at the fixture's `NOW`. Each task gets a fresh copy of its fixture's state.
- Invalid arguments return a structured error that a model can read and recover from.
- Tool schemas are generated from code, never hand-edited.
- The acting user comes from the case's `user_id`, passed as "Current user: <id>" in the system prompt.
- `ops` is fully simulated, never a real cluster. Each revision in the fixture is marked healthy or broken, pod status follows the current revision, and logs are canned per revision, built from Loghub samples so they look real. `rollback` and `scale` change state deterministically.
- Cross-API rule: any change in the `prod` namespace requires `change_ticket`, the ID of an open helpdesk ticket. Without it, return an error that explains the rule.
- Chaos (`gym_env/chaos.py`): a seeded fraction of calls return a transient 503. Off by default; on only in the demo or in runs flagged `--chaos`.

## Seatbelt (`common/seatbelt.py`, used by `common/tool_loop.py`)

- Before executing a call, the tool must exist and its arguments must validate against the tool's JSON schema. If not, send the validation error back to the model as the tool result.
- Environment errors are sent back to the model as tool results.
- Invalid calls and errors count toward a limit of 2 retries per step. Transient 503s are retried automatically with backoff, up to 3 times, without involving the model.
- When the model gives its final answer for a step, it runs a self-check: a structured `{complete: bool, reason}` answer to "given the request and the calls made, is the task complete?". If `false` and retries remain, the step continues; otherwise the step fails.
- A failed step fails the case, except in `team_escalate`, where it goes to escalation.
- Every seatbelt event is logged in the trace.

## Escalation

- Lightning receives the step's conversation so far, including calls already executed and their results, and finishes the step with the same tools and seatbelt. It never restarts the step, so no action is repeated.
- The step is marked `escalated: true` and its tokens are priced at Lightning's rate.

## Scoring

- `gym_env/verify.py` is the only scoring logic. For each case: run `gold_calls` on a fresh fixture to get the expected state, run the system on another fresh fixture, and pass only if the final states match exactly. `read_only` cases must also contain every `answer_must_contain` string in the final answer (a substring check that ignores case).
- Single-model setups run as NeMo Gym agents. The team runs through Gym's remote-agent integration if that works within its time box; otherwise `eval/run_eval.py` drives the team against the same resources server and verifier.
- Each case runs 3 times and passes if at least 2 runs pass. Report fractions like `37/50`, never percentages.
- Team failures are labeled `routing` if the plan's agents don't match the APIs in `gold_calls`, otherwise `specialist`.
- Eval runs refuse to start if chaos is on (unless flagged `--chaos`) or if `LLM_MODE=replay`.
- Cost: each model has a price per 1M tokens in `configs/eval.yaml`, derived from the GPU's hourly price and measured throughput, with the derivation noted. Every setup is priced with the same formula.
- The factory's filter uses the same verifier: a Lightning rollout becomes training data only if it passes and contains no failed calls.

## Tool-call format

- Render training examples with the student base model's own chat template (`tokenizer.apply_chat_template(..., tools=...)`) so fine-tuned outputs match what vLLM's tool parser expects.
- After every training run, run `scripts/smoke_toolcall.py --model <adapter>`. It must parse at least 19/20 calls. If it doesn't, stop and fix the format before doing anything else.

## Commands

Planned. Create these as you build, and keep this list accurate.

```
uv sync                                     # Windows: core deps only. Linux: also NeMo Gym
uv sync --group train                       # Brev box: adds TRL, PEFT, transformers
uv venv .venv-vllm --python 3.13            # Brev box: vLLM in its own venv
uv pip install --python .venv-vllm/bin/python vllm==0.27.1 --exclude-newer 2026-10-03T00:00:00Z
scripts/serve_lightning.sh                  # background, port 8000; log and PID in runs/vllm/
scripts/serve_student.sh                    # base model + every adapter in adapters/, port 8001
scripts/stop_vllm.sh [lightning|student]    # no name: stop both
uv run python scripts/smoke_toolcall.py --model <name>   # adapters: add --min-parsed 19
LLM_MODE=record <command>                   # also save every common/llm.py response to runs/cache/
LLM_MODE=replay <command>                   # answer from runs/cache/ with no server (never for evals)
uv run python -m gym_env.fixtures.make_fixtures --config configs/fixtures.yaml   # logs.json + incident fixtures
uv run python -m gym_env.tasks --cases <cases.jsonl> --output <rows.jsonl>       # our cases -> Gym task rows
# Gym servers (itops + simple agent + Lightning as policy). UV_EXCLUDE_NEWER keeps Gym's per-server
# venvs (built under runs/gym_venvs/) within rule 1. Runs in the foreground; Ctrl+C stops it.
UV_EXCLUDE_NEWER=2026-10-03T00:00:00Z uv run gym env start --resources-server itops --model-type vllm_model \
  --model lightning --model-url http://localhost:8000/v1 --model-api-key EMPTY +uv_venv_dir=runs/gym_venvs
uv run gym eval run --no-serve --agent itops_simple_agent \
  --input resources_servers/itops/data/example.jsonl --output runs/gym_smoke/rollouts.jsonl
uv run python -m eval.run_eval --setup base_seatbelt --api helpdesk --split practice
uv run python -m factory.run_round --api ops --round 1
uv run python -m eval.run_eval --setup team_trained --split practice
uv run python -m bench.run --suite all
scripts/demo_reset.sh
uv run python -m eval.report --run <run_id>
uv run pytest
```

## Conventions

- Small, plain modules. Type hints everywhere, pydantic for data formats.
- All model calls outside NeMo Gym go through `common/llm.py`, which logs tokens and latency to the active trace. It supports `LLM_MODE=live|record|replay`: `record` saves responses to `runs/cache/`, and `replay` serves them without a GPU, for building the team and demo, and as a demo fallback. Never evaluate in replay mode.
- Every script takes `--config`, and writes its resolved config and the git commit hash into its run folder.
- Fix seeds. Eval temperature lives in `configs/eval.yaml`.
- Secrets go in `.env` (never committed).

## Working style

- For anything bigger than a small fix, propose a short plan and wait for approval.
- After finishing a task, run its check and show the output.
- When unsure about a library API (NeMo Gym, vLLM, TRL, Unsloth), read the installed version's docs or source instead of guessing.
- Update `PLAN.md` checkboxes and the Commands section when things change.
- Log every command you run on the box in `COMMANDS_LOG.md`, under the current task, with a comment saying what it was for (and its result when that matters).
