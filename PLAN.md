# PLAN.md: Small-model IT operations team

The hackathon runs October 17–18, 2026 at the Red Hat Annex, Raleigh. All code must be written during the event. Fill these in during the first hour: `<STUDENT_ID>` (student base model), the Brev GPU type, and the submission deadline.

**Deadline:** submission and presentation on Sunday, October 18, at 11:00 a.m. ET. Feature freeze is Sunday at 07:00 ET.

**GPU:** one H200 (141 GB) on Brev. **Team:** solo.

**Student base:** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (3.97B). See task 0.2 and `configs/models.yaml`.

Repo: [`techsachinkr/small-model-itops-team`](https://github.com/techsachinkr/small-model-itops-team), private until submission. Use this name in the README, the Hugging Face dataset and model cards, and the submission form.

## How to use this with Claude Code

1. Put `CLAUDE.md`, this file and `PROMPTS.md` in the repo root. The file must be named exactly `CLAUDE.md`; Claude Code loads it automatically when you start `claude` in that folder.
2. Add `.claude/settings.json` so Claude Code's file tools can't read or edit the test set:

   ```json
   {
     "permissions": {
       "deny": ["Read(./data/test/**)", "Edit(./data/test/**)"]
     }
   }
   ```

   This doesn't block shell commands like `cat`, so the rule in CLAUDE.md still matters.
3. For each task, follow the loop in `PROMPTS.md`: run `/clear`, press Shift+Tab for plan mode, start the task, review the plan, implement, verify, commit.
4. If a task runs past its time box, check the cut list at the bottom before pushing on.
5. Long GPU jobs (training, Lightning rollout collection, eval runs) are run by you in your own terminal, not by Claude in the background (CLAUDE.md "Working style", decided 2026-10-17). Where a task prompt below says "run" or "start" one of these, Claude prepares the command and you run it.

## What we're submitting

- **Main track:** Multi-Agent Systems. The team beats one small model with every tool.
- **NeMo Gym bounty:** our IT-ops environment in NeMo Gym, and baseline vs final scores for small specialists on held-out tasks, with each improvement isolated (seatbelt, round 1, round 2).
- **Nemotron Lightning bounty:** Lightning is the teacher, orchestrator, failure analyst and escalation target. We measure its contribution as planner (vs a small planner), as teacher (verified data yield), and through escalation.
- **Red Hat vLLM bounty:** vLLM serves Lightning and every specialist adapter. We report measured results for multi-LoRA serving vs separate servers, concurrency, prefix caching, structured outputs, and speculative decoding if supported.

Each bounty needs a short written explanation at submission (task 5.2).

## Event requirements (copied 2026-10-17)

Source: `event_req/NVIDIA x Red Hat Small Models Hack.pdf` and the Submission Checklist and Demo Video Instructions pages it links to.

**Submission:** through the [Airtable form](https://airtable.com/appm7NS9oN8K3friP/pagqXe6ElIlXx6oa3/form), due Sunday, October 18, at 11:00 a.m. ET. One submission per team. Required:
- Team name, project description, and the challenge selected (Multi-Agent Systems).
- A 3–5 minute demo video, unlisted on YouTube or Vimeo, showing the core loop live.
- Repo link (public, or invite the judges). The README must have: a quick start (commands to run); the tech stack and an architecture diagram; how to reproduce the demo (env vars, API keys, a sample `.env`); the datasets and synthetic data used, with provenance; known limitations and next steps.
- A deployed URL if there is one, otherwise a short screen capture of the working app.
- Team roster: names, roles, contacts.

**Demo video:** 3–5 minutes, a screen recording with the camera on (Loom suggested). Recommended flow:
1. Introduce the team (20–30 s).
2. Elevator pitch: what we built, how we approached it, why it's exciting (30–40 s).
3. Live demo of the core loop, input → processing → output, with as few cuts as possible (45–60 s).
4. How we built it: architecture and stack, which models and why, agents, environments and evaluators, how the hardware shaped the design, bottlenecks, tradeoffs and workarounds (60–90 s).
5. Why it matters: answer "so what?" (20–30 s).

**Bounties**, each with a short written explanation:
- **Nemotron Lightning:** Lightning must be a core part of the project. Show its contribution (quality, speed, efficiency, usefulness or creativity). Submit what Lightning does, why we chose it for that part, and how we optimize its capabilities.
- **NeMo Gym:** a Gym environment with tasks, tools the agent can use, and code that scores each attempt. Show baseline vs final scores on the same held-out tasks and eval configuration, isolating the improvements. Submit the Gym setup, baseline vs final results, how to run scoring, and how we made the improvements.
- **Red Hat vLLM:** vLLM as the serving layer for a core model, working and central to the build. Show what we got from it (performance, efficiency, throughput, latency, scalability or reliability). Submit how vLLM fits the architecture, how we configured and optimized it, and measurable results. "Depth over breadth."
- **NVIDIA Developer Champions Choice Award:** chosen by NVIDIA Developer Champions; no extra requirements.

**Judging:** the page still shows only criterion 1, Technical Execution & Completeness: a working end-to-end system without major failures, meaningful use of open data, and technical depth. For Multi-Agent Systems, that means several small models that collaborate, divide responsibilities and exchange information to do what one model couldn't easily do.

**The 33B rule as written** says it "applies to the total params of all generative language models", which can be read as a combined total (Lightning 30B + Nano 4B 3.97B = 33.97B). The organizers answered per model (task 0.1, Q1). Keep a link to or screenshot of that answer, and cite it in the README.

What this changes in our plan:
- README (task 5.2) also needs a quick start, a sample `.env`, next steps and the team roster.
- The demo script (task 5.3) is the live-demo and build-narration part of the recommended flow. Add the team intro, pitch and "so what", for 3–5 minutes in total.
- vLLM is judged on depth over breadth, so in task 4.1, do (1) multi-LoRA vs separate servers and (2) concurrency thoroughly before anything else.
- The repo is private: make it public, or invite the judges, before submitting.

---

## 0. Right now (first hour)

### [ ] 0.1 Ask the organizers (Discord #small-models-hack-raleigh, or in person)

1. Is the 33B limit **per model**, or the **combined total** of every generative model in the system? This decides the architecture.
2. Do models used only as comparison baselines count toward the limit?
3. Can one project win a track and bounties at the same time?
4. When exactly is the submission deadline, and what are the full judging criteria? (The page we saved only shows criterion 1.)
5. Can we train on public datasets that were generated by models over 33B (for example the `tool_calling` split of `nvidia/Nemotron-Post-Training-Dataset-v1`, generated by DeepSeek-R1 and Qwen3-235B)? Until we hear back, outside datasets are for evaluation only.

Answers (2026-10-17):
- **Q1:** Per model, not combined. Each model must be ≤33B. The organizers put this in terms of fine-tuning; CLAUDE.md rule 2 still applies the limit to every generative model. Lightning (about 30B) and the student can both be used, so the "If the limit is combined" options below aren't needed.
- **Q5:** Yes. Training on public datasets generated by models over 33B is allowed.
- **Q4 (partly):** the Submission Checklist confirms the deadline, Sunday, October 18, at 11:00 a.m. ET. The judging section still shows only criterion 1 (see "Event requirements").
- Q2, Q3: still open.

**If the limit is combined:** Lightning alone is about 30B, so check its official count on the model card. If it's 30B, a student of 3B or less could still fit, with no separate draft model for speculative decoding. If it doesn't fit, choose one:
- (a) Drop Lightning and use a smaller Nemotron as teacher and orchestrator. This is safer but gives up the Lightning bounty.
- (b) Make Lightning the only model, with specialists as LoRA adapters on Lightning itself. This is more Lightning-centric, but only viable if a quick test shows LoRA training and vLLM LoRA serving work for Lightning.

### [x] 0.2 Set up

- [x] Open the event's Submission Checklist and Demo Video Instructions, and copy any requirements (video length, required links, repo visibility) into this file. See "Event requirements" near the top.
- [x] Team captain redeems the Brev code (Billing → Redeem Code) and invites teammates (Team → Generate Invite Link). Share SSH access on the instance.
- [x] Pick the GPU. Lightning in BF16 needs roughly 60+ GB before any other model. Prefer a single GPU with 141 GB or more, or two GPUs. On Blackwell GPUs, Lightning's NVFP4 build is much smaller. Record the plan in `configs/gpu.md`.
- [x] Read Lightning's model card for the official parameter count, license, recommended vLLM settings and tool-call parser.
  - 30B total, 3B active (card). Hugging Face's safetensors metadata reports 31.6B (32.9B summed by dtype), likely including the MTP layers; every reading is ≤33B, and rule 2 uses the card's number.
  - License OpenMDW-1.1. Tool-call parser `qwen3_coder`, reasoning parser `nemotron_v3`. Sampling: temperature 1.0, top_p 0.95; thinking on by default (`enable_thinking`).
  - vLLM 0.27.1, as pinned by the card (on PyPI 2026-08-11, before the rule 1 cutoff). It needs torch 2.13.0 while the train group uses 2.14.1, so it lives in `.venv-vllm/`. The newest allowed release is 0.30.0 (2026-09-22); it adds Nemotron-H MTP fixes, worth a look only for speculative decoding in 4.1.
  - The card's 1×H100 recipe, our starting point: `--max-num-seqs 128 --enable-prefix-caching --async-scheduling --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder --enable-auto-tool-choice`.
- [x] Choose the student base (1–4B) with LoRA support in vLLM and a working tool-call parser. A small Nemotron model is preferred if one qualifies.
  - **`nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`**: 3.97B (card), NVIDIA Nemotron Open Model License. It's the same `nemotron_h` hybrid family as Lightning, and its chat template uses the same tool-call format: `qwen3_coder` tool parser, built-in `nemotron_v3` reasoning parser (the card's `nano_v3` plugin isn't needed). Shortlisted alternatives: `Qwen/Qwen3-4B-Instruct-2507` (fallback, `hermes` parser) and `ibm-granite/granite-4.0-micro` (`granite4`).
  - LoRA check on vLLM 0.27.1 (commands and results in `COMMANDS_LOG.md`): untrained PEFT adapters load, and vLLM reproduces HF+PEFT greedy output exactly (32/32 tokens) for attention + MLP and for Mamba `in_proj`; a zero-init adapter matches the base. PEFT refuses Mamba `out_proj` and `conv1d` on `nemotron_h`. A tool call parsed with thinking both on and off.
  - For 2.2: LoRA targets `q_proj k_proj v_proj o_proj up_proj down_proj in_proj`. The `mamba_ssm` and `causal_conv1d` kernels aren't installed, so HF training uses the slower plain-PyTorch Mamba path. Recommended temperature differs from Lightning's: 0.6, top_p 0.95 for tool calling.
  - For the serve scripts: put `.venv-vllm/bin` on PATH (FlashInfer runs `ninja` from PATH when it JIT-compiles at startup). vLLM leaves prefix caching off by default for this model (relevant to 4.1).
- [x] Decide whether `calendar` is in scope. The default is no: two specialists (helpdesk and ops), with calendar as a stretch goal.
  - Decided 2026-10-17: out of scope. Two specialists, helpdesk and ops; calendar stays a stretch goal.
- [x] Split roles (see Team split). Person C starts writing the test set immediately.
  - Solo, so there's nothing to split: follow the "Solo" line under Team split. The test set is still written by a human only (CLAUDE.md rule 4).

---

## Design decisions

**helpdesk.** Ticket IDs look like `HD-1042`, priority must be `P1`–`P4`, and hardware tickets need a `device_id`, so the model must look it up first.
- `list_devices(employee_id)`
- `create_ticket(employee_id, category[hardware|software|access|network|incident], summary, priority, device_id?)`
- `list_tickets(employee_id, status?[open|in_progress|resolved])`
- `get_ticket(ticket_id)`
- `update_ticket(ticket_id, status?, priority?, note?)`
- `reset_password(employee_id, delivery[email|sms])`

**ops.** Namespaces are `prod` and `staging`. Pod names are generated (like `payments-7d9f-x2k`), so reading logs needs a pod lookup first. Rollbacks need a revision number from the history. Any change in `prod` needs `change_ticket`, the ID of an open helpdesk ticket.
- `list_deployments(namespace)`
- `get_pods(namespace, deployment)`
- `get_logs(namespace, pod, tail?)`
- `rollout_history(namespace, deployment)`
- `rollback(namespace, deployment, to_revision, change_ticket?)`
- `scale(namespace, deployment, replicas, change_ticket?)`

**calendar (stretch).** Slot IDs are opaque and only come from `find_slots`, so booking always takes two steps.
- `find_slots(resource[repair_desk|room_a|room_b|war_room], date, duration_min)`
- `book_slot(slot_id, employee_id, purpose)`
- `list_bookings(employee_id, date?)`
- `cancel_booking(booking_id)`

**Fixtures.** `default` has 12 employees, 15 devices, 10 existing tickets, and `prod` and `staging` namespaces with 5 healthy deployments each. `make_fixtures.py` creates seeded incident variants, for example a bad deploy that breaks one deployment. Logs for broken revisions are built from Loghub samples. `NOW` is a fixed Monday at 09:00.

**Open data.** We use four sources. The outside ones are listed in `data/open/DATA_SOURCES.md` with their licenses, along with the sources we considered and rejected:
- Loghub system logs, to make ops logs realistic. Loghub's license allows research or academic use with attribution and is not Apache-2.0, so log lines derived from it keep Loghub's license.
- The NeMo Gym Workplace Assistant dataset (`nvidia/Nemotron-RL-agent-workplace_assistant` on Hugging Face), as a general tool-use check before and after fine-tuning.
- NVIDIA's When2Call (`nvidia/When2Call` on Hugging Face), to check that the student still knows when not to call a tool.
- Our own synthetic tasks and verified Lightning trajectories, published on Hugging Face with their methodology.

Outside datasets are for evaluation only for now. The organizers allow training on data generated by models over 33B (task 0.1, question 5), but any outside training data would be a separate ladder row, not part of `r1`.

**Demo request.** "The payments app keeps crashing after this morning's deploy: roll it back and open a P1 ticket." If calendar is in scope, add "and book a postmortem slot for tomorrow." The ticket must exist before the `prod` rollback.

**Data sizes** (smaller than originally planned, because time is short):
- Practice set: 12 single-API cases per API plus 8 multi-API cases. Claude Code may help draft these; a human reviews every one.
- Test set: 10 per API plus 15 multi-API cases. Written by humans only. About 20% should be `no_tool` or `missing_info`, including risky requests (like "roll back everything in prod") where the right move is to refuse or ask. Add a few `read_only` cases too.
- Factory round 1: about 1,000 kept trajectories per API (about 80% `sampled`, 20% `written`). Round 2: about 300 targeted (`written`).
- LoRA starting point: r=16, alpha=32, learning rate 2e-4, 2 epochs, loss on assistant turns only.
- Seatbelt: 2 retries per step, plus up to 3 automatic retries for transient 503s.

---

## Time budget

Count hours from the start of building (Saturday late morning). Replace "Deadline" with the real submission time.

| Block | When | Goal |
|---|---|---|
| 0. Setup | Hour 0–1 | Questions answered, GPU running, models chosen |
| 1. Environment and baseline | Hours 1–5 | Gym environment works; baseline scores for both APIs |
| 2. Factory round 1 | Hours 5–10 | Lightning rollouts collected; training starts before sleep |
| Overnight | | Round 1 training and evals run unattended |
| 2–3. Round 1 results, team | Sunday morning | `fallback-v1` tagged first thing; orchestrator working |
| 4. vLLM benchmarks | In parallel (person B) | Serving results table |
| Feature freeze | Sun 07:00 ET (deadline minus 4 h) | Only fixes, final eval, writing, video |
| 5. Final eval and submission | Sun 07:00–11:00 ET | Test run, README, bounty write-ups, video, submit |

## Team split

With three people:
- **A (factory and training):** 2.1–2.3, 3.2 (`single_trained`)
- **B (serving, team, demo):** 0.2 serving, 3.1, 3.3, 4.1, 5.3
- **C (environment and eval):** 0.1 skeleton, 1.1, 1.2, the test set, 5.1, 5.2

Each person runs their own Claude Code session on their own branch. The data formats in CLAUDE.md are the contract between you. B can build with `LLM_MODE=replay` while the GPU is busy. Agree on a GPU schedule, because serving and training compete for memory.

Solo: stages 0–2 are the core, and stop after round 1 if needed. Then add the orchestrator with untrained and trained specialists, and spend the last block on benchmarks and submission.

---

## Stage 0: Skeleton and serving

### [x] 0.1 Repo skeleton

```
Read CLAUDE.md and PLAN.md. Create the repo skeleton from "Repo layout" using uv
(Python 3.11): pyproject.toml with nemo-gym, vllm client deps, trl, peft and
pydantic; empty modules each with a one-line docstring; .gitignore (adapters/,
runs/ except runs/final/, .env); configs/models.yaml, configs/eval.yaml and
configs/gpu.md with commented placeholders; data/open/DATA_SOURCES.md with a
table (source, link, license, how we use it); and common/schemas.py implementing
every data format in CLAUDE.md as pydantic models. No other logic yet. Show me
the tree.
```

**Done when:** `uv run python -c "import common.schemas"` works and the tree matches CLAUDE.md.

### [x] 0.2 Serving, the LLM client and replay mode

```
Fill configs/models.yaml with Nemotron 3.5 Lightning and Nemotron 3 Nano 4B, including
each model's official total parameter count from its model card, its vLLM
tool-call parser (check the installed vLLM docs and Lightning's model card),
and its port. Record the GPU memory plan in configs/gpu.md. Write
scripts/serve_lightning.sh, scripts/serve_student.sh (LoRA enabled, loading any
adapters in adapters/) and scripts/stop_vllm.sh. Put a price per 1M tokens for
each model in configs/eval.yaml, with a note on how it was derived. Write
common/llm.py: one function that takes a model name, messages, tools and an
optional JSON schema for structured output, calls the right server through the
OpenAI client, and returns the response plus prompt tokens, completion tokens
and latency. Support LLM_MODE=live|record|replay as in CLAUDE.md. Then write
scripts/smoke_toolcall.py: 20 prompts with a dummy weather tool, reporting how
many came back as parsed tool calls.
```

**Done when:** both servers run on the GPU together, `smoke_toolcall` shows at least 18/20 parsed calls for each model, and a recorded call replays correctly with vLLM stopped.

Notes for later tasks (from 0.2; details in `configs/gpu.md` and `COMMANDS_LOG.md`):
- 1.2: `llm.chat()` returns tokens and latency but doesn't write a trace; `tool_loop.py` should add them into each `Step`/`Trace`.
- 1.2: vLLM returns reasoning as `message["reasoning"]`, but both chat templates read `reasoning_content` from history. Rename the field (or drop it) when sending an assistant turn back.
- 1.2: structured output works with thinking on for both models: the schema constrains only the answer, after the reasoning.
- 4.1: the student's `--max-num-seqs` is 256 because its 0.15 memory share holds 643 Mamba cache blocks (one per running sequence). Prefix caching is on for Lightning and off for the student (vLLM's default for it).

## Stage 1: Environment and baseline (NeMo Gym)

### [x] 1.1 The NeMo Gym environment

```
First read the installed NeMo Gym docs: the Stateful Environment tutorial, the
Execution and State Match verification pattern, and how a resources server is
configured and run. Summarize the minimum we need in 10 lines and wait for my
OK. Then build gym_env/ as one resources server following "Environment rules"
in CLAUDE.md, with the helpdesk and ops tools and quirks from PLAN.md "Design
decisions": isolated state per task, the change_ticket rule, chaos (off by
default), and gym_env/verify.py implementing "Scoring". Download a small Loghub
sample into data/open/ (record it in DATA_SOURCES.md) and use it for the canned
logs of broken revisions. Write fixtures/default.json and
fixtures/make_fixtures.py, and gym_env/tasks.py to convert our cases to Gym's
format. Write pytest tests for: determinism, readable errors for invalid
arguments, the change_ticket rule, seeded chaos, and the verifier passing gold
calls and failing a wrong final state.
```

**Done when:** pytest passes, and Gym can run one hand-made task end to end against the resources server.

Notes for later tasks (from 1.1; commands and results in `COMMANDS_LOG.md`):
- Decisions: Gym only finds servers at `resources_servers/<name>/`, so `resources_servers/itops/` is a thin wrapper around `gym_env/server.py`. The verifier compares the session's final state, not a replay of the model's calls. Ticket `summary` and `note` only have to be present (option B); per-case keywords (option C) can be added later.
- "Open ticket" for `change_ticket` means unresolved: status `open` or `in_progress`.
- 1.2: tool errors come back as HTTP 400/404 with `{"error": {"code", "message"}}`, and chaos as 503 with `"retryable": true`, so the seatbelt can tell them apart. `gym_env.env.Env` runs the same tools without HTTP. Reuse `gym_env.tasks.system_prompt` so every setup sees the same prompt. `max_steps: 10` in `resources_servers/itops/configs/itops.yaml` is only for smoke runs.
- 1.2: cases must pin down every structured field the gold checks. In the first smoke run, Lightning failed only the demo request, filing the ticket as `software` where the gold says `incident`. After adding category definitions to `create_ticket`'s schema it chose `incident` in 2 of 3 runs.
- 1.2: `answer_must_contain` ignores case (decided 2026-10-17, after a correct "Status: Open" answer failed against "open" in the rerun).
- `gym eval run --input` needs `--no-serve` and servers started with `gym env start` (Commands in CLAUDE.md).

### [x] 1.2 Seatbelt, tool loop and the baseline

```
Write common/seatbelt.py exactly as in CLAUDE.md, and common/tool_loop.py that
runs a model through the seatbelt against our tools. Integrate it as a NeMo Gym
agent (see the Gym docs on integrating existing agents) with a switch to turn the
seatbelt off. Write gym_configs/ for the setups base_raw, base_seatbelt and
lightning, sharing one eval config so that only the model and seatbelt switch
differ. Write eval/run_eval.py wrapping Gym evaluation for --split practice,
with the refusal rules from CLAUDE.md, and a guarded test path that refuses to
run without --final. Results go to runs/<run_id>/. Add pytest tests: the
seatbelt recovers from 503s, sends back invalid arguments, and respects the
retry limit. Then help me draft 12 practice cases each for helpdesk and ops.
```

**Done when:** a human has reviewed the practice cases, and a practice table exists for `base_raw`, `base_seatbelt` and `lightning` on both APIs. If the base model already scores 11/12 or better on an API, make that API harder before going on.

Practice baseline (2026-10-17; `runs/practice_baseline.md`, built by `eval.report` from `runs/baseline_*`; 36/36 attempts each; thinking off):

| setup | helpdesk | ops |
|---|---|---|
| `base_raw` | 4/12 | 8/12 |
| `base_seatbelt` | 4/12 | 8/12 |
| `lightning` | 9/12 | 7/12 |

Notes for later tasks (from 1.2; commands and results in `COMMANDS_LOG.md`):
- Every setup runs as the Gym agent `responses_api_agents/itops_agent` (code in `common/gym_agent.py`) around `common/tool_loop.py`. `eval/run_eval.py --setup <s> --api <a> --split practice` starts and stops the Gym servers itself; the setup's vLLM server must be up and port 11000 free. New setups are a `gym_configs/<setup>.yaml` that sets only the model (and the seatbelt switch).
- Eval settings (`configs/eval.yaml`): temperature 0.6, top_p 0.95, thinking off, max_tokens 8192, 10 turns per step. Thinking off because with it on the student already passed 10–11/12 per API (no room for the ladder); off uses about a third of the tokens and half the latency (`runs/think_*`).
- 2.1: factory rollouts aren't an eval setup. Collect them with Lightning's thinking on: with it on, the verifier passed 33/36 (helpdesk) and 30/36 (ops) of Lightning's practice attempts, against 27/36 and 21/36 with it off. Render `text` the way the student is evaluated, thinking off: drop the reasoning and use the chat template with `enable_thinking=False`.
- 2.1: the filter keeps a rollout only if `reward` is 1 (a step the seatbelt gave up on scores 0; `verifier_reward` keeps the state match) and no call failed. In traces, call status 0 means blocked by the seatbelt.
- The seatbelt self-check asks "is there a tool call you still need to make now?" (`{reason, tool_call_needed}`), with thinking off. Asked "is the task complete?", Lightning said no whenever it had correctly asked the user, and the seatbelt discarded correct attempts. In the baseline it rescued 3 attempts and broke none; once, before the baseline, it pushed a correct question into a wrong action (`pr-hd-09`), so keep watching it.
- `base_seatbelt` = `base_raw` on practice cases (attempts: 16 vs 14 helpdesk, 23 vs 23 ops). With thinking off the student rarely makes invalid calls; the seatbelt's value will show under chaos and as the trigger for escalation (3.3).
- Both vLLM servers emit compact JSON for structured outputs (`--structured-outputs-config '{"backend": "xgrammar", "disable_any_whitespace": true}'`); without it a self-check answer ran into whitespace until max_tokens.
- Verifier (after the evaluator review): tickets the gold creates or updates need at least the gold's number of notes; optional `answer_must_not_contain` per case.
- Deferred from the review: a new fixture (not edits to `default`) with several tickets per user and real change tickets, for ticket picking and the multi-API cases in 3.1; a check that prod change tickets come from the request or a tool result; coverage gaps (successful email reset, staging rollback, an ops decline case, logs-then-decide).
- A stray `gym env start` stops only with Ctrl+C for its whole process group: `kill -INT -- -<pgid>`.

## Stage 2: The factory with NeMo Gym

This stage completes the fallback project: a NeMo Gym bounty entry plus a Fine-Tuned result.

### [x] 2.1 Tasks and Lightning rollouts

```
Implement factory/sampler.py and factory/writer.py per CLAUDE.md: the sampler
picks valid actions from the fixtures and computes gold_calls in code for each
kind (single with one and two steps, no_tool, missing_info); the writer phrases
them as natural requests (origin=sampled) and writes about 20% entirely with
Lightning (origin=written). Save tasks with their generation config and seed.
Implement factory/collect.py to run Lightning on these tasks as NeMo Gym
rollouts, and factory/filter.py to keep only rollouts the verifier passes with no
failed calls, rendering "text" with the student's chat template. Also write
eval/decontaminate.py to drop tasks too similar to practice cases (--against
practice only for now). Run 50 tasks for helpdesk and show me the stats (kept
vs rejected, by origin and reason) and 5 kept examples.
```

**Done when:** a human has read 10 kept examples and they look right. Then start full round 1 for both APIs (about 1,000 kept each).

Pilot (2026-10-17; `data/generated/helpdesk/pilot/`, run `pilot_helpdesk`; commands and results in `COMMANDS_LOG.md`): 50 helpdesk tasks (40 sampled, 10 written), 44 after decontamination, 40 kept (sampled 31/35, written 9/10; rejected: 2 state_mismatch, 2 self_check_nudge). The human read 10 kept examples: they look right, except that a no_tool answer offering to look up a colleague's mobile number passed (nothing changed), so the `colleague_private_info` template is gone. The pilot isn't part of round 1.

Notes for round 1 and later tasks:
- Settings in `configs/factory.yaml`. Collection is `configs/eval.yaml` with thinking on and one attempt per task (temperature stays 0.6, where Lightning's practice pass rates were measured). Writing whole tasks uses thinking on (off, 4 of 8 trial tasks contradicted the fixture); phrasing uses thinking off.
- Decontamination has three rules (text 0.7; same user, fixture and kind 0.25 for tasks that change nothing; the same changes as a practice case or part of one). It dropped 6 of 50 in the pilot; ops drops more (about 13% in calibration), because rollbacks on incident fixtures often land on a practice case's answer.
- `text` uses ASCII hyphens and spaces in Lightning's answers (it writes IDs like "HD‑1041" with U+2011); `messages` keeps Lightning's own text.
- The self-check sometimes says a tool call is needed when Lightning correctly asks the user; those rollouts are rejected (2 of 7 missing_info/no_tool tasks in the pilot).
- Round 1 size: 1,300 helpdesk and 1,500 ops task slots (`--name round_1`), for about 1,000 kept each: the pilot kept 40 of 50 slots, and ops loses more to decontamination. If either API ends under about 1,000, add a top-up batch with another `--name` and seed. Collection run ids `round1_helpdesk` and `round1_ops`. Task mix, files and methodology: `data/generated/README.md`.
- Round 1 training data (2026-10-17; runs `round1_helpdesk` and `round1_ops`, commit `49a5232`; `data/generated/<api>/round_1.jsonl` and `.stats.json`): **helpdesk 1,062** kept of 1,198 rollouts (sampled 876, written 186; single 855, no_tool 101, missing_info 80, read_only 26), **ops 1,016** of 1,149 (sampled 871, written 145; single 696, no_tool 169, missing_info 121, read_only 30). Both reach the ~1,000 target, so no top-up. Main rejections: helpdesk 64 state_mismatch (Lightning guessed a device in which_device, or a different update), 31 self_check_nudge, 29 answer_mismatch; ops 48 failed_call (46 in unknown_deployment: Lightning tries the call, gets the 404, then asks), 33 state_mismatch (15 guessed a namespace in which_namespace), 32 self_check_nudge.
- 23 of helpdesk's 29 answer_mismatch rejections were right answers that the verifier's substring check missed because Lightning wrote look-alike characters ("HD‑1034" with U+2011, "in progress" with U+202F); they were mostly the written read_only tasks (26 of 55 kept). In the task 1.2 practice baselines (thinking off) this cost one attempt (Lightning, pr-ops-08). Whether `gym_env/verify.py` should map look-alike hyphens and spaces to ASCII before its answer checks is open (a scoring change for every setup).
- Round 1 tasks: helpdesk 1,289/1,300 written in 2,664 s, 1,198 after decontamination; ops 1,472/1,500 in 2,864 s, 1,149 after decontamination (22% dropped, mostly the changes rule: pr-ops-05, 03, 06). If the ops filter keeps under about 1,000, top up with another batch (`--name round_1b`, another seed). Both configs record commit `365ae42` (read at the end); the code that ran is `8e036d4` (only docs differ); the writer now reads git at the start.
- Written tasks that hit `max_tokens_write` (12,288) while still thinking come back empty ("the answer isn't a JSON object"): 63 of 355 ops calls and 16 of 386 helpdesk calls in round 1. For later rounds, raise it (the context is 32k; prompts are about 6-8k tokens) or accept the retries.
- Writing round 1 (both APIs at once, writer concurrency 16 each), measured 21 minutes in: written tasks averaged about 5,700 completion tokens per call, 22% of those calls were retries, and Lightning generated about 1,450 tokens/s with its KV cache near 12%. So a higher `writer.concurrency` is safe.

### [ ] 2.2 Training and the round 1 evaluation

```
Write factory/train_lora.py using TRL and PEFT (or Unsloth per Gym's tutorial)
with the LoRA defaults in PLAN.md, saving adapters to adapters/<api>_r1/ with
config and metrics in runs/. Follow configs/gpu.md for pausing servers. Restart
serve_student.sh with the adapters, run smoke_toolcall on each, and run the r1
setup on practice next to base_raw, base_seatbelt and lightning.
```

**Done when:** both adapters pass the smoke check (at least 19/20), and the practice ladder table has an `r1` row. Commit and tag `fallback-v1`.

### [ ] 2.3 Failure analyst and round 2

```
Write factory/analyst.py: read the failed practice traces for one adapter and
have Lightning, with structured output, group the mistakes into clusters (name,
description, case_ids, guidance) saved as JSON. Add a targeted mode to
writer.py that writes new tasks following each cluster's guidance without
copying practice requests. Wire everything into factory/run_round.py --api <api>
--round <n>. Run round 2 for whichever API has more room to improve, and add an
r2 row to the ladder.
```

**Done when:** the ladder has an `r2` row for at least one API. Round 2 practice scores are optimistic because the analyst saw practice failures; the honest numbers come from the test set.

## Stage 3: The team

### [ ] 3.1 Specialists and the Lightning orchestrator

```
Write agents/specialist.py: wraps common/tool_loop.py (seatbelt included) with
one adapter (or the base model) and one API's tools. Write a JSON card per
specialist in agents/cards/ (A2A Agent Card fields: name, description, skills)
and agents/registry.json listing which specialists are active. Write
agents/orchestrator.py: load the active cards, ask Lightning for a JSON plan
via structured output, with steps {agent, instruction, depends_on}; run steps
in dependency order, pass earlier results (like a ticket ID) into later
instructions, stop after 6 steps, and say clearly when no specialist covers part
of a request. Add the routing vs specialist failure labels from CLAUDE.md. Then
help me draft 8 multi-API practice cases. Time box: if running the team as a
NeMo Gym remote agent isn't working within 2 hours, have eval/run_eval.py drive
the team against the same resources server and verifier instead.
```

**Done when:** the demo request works end to end (ticket first, its ID passed to ops, pods back to Running), and `team_base` and `team_trained` are scored on the multi-API practice cases.

### [ ] 3.2 System baselines

```
Add the setups single_base, single_lightning and single_trained. For
single_trained, train one adapter on the union of all kept training examples
(same examples, same hyperparameters) and serve it alongside the specialists.
All three run one model with every tool and no orchestrator, under the shared
eval config. Run every system setup on the multi-API practice cases and print a
table with correct cases, mean latency and cost per 1,000 requests.
```

**Done when:** the system practice table has all available setups.

### [ ] 3.3 Escalation and Lightning's planner contribution

```
Implement escalation exactly as in CLAUDE.md, add the setup team_escalate, and
report escalated steps, the share of steps handled by small models, and cost
per 1,000 requests. Add the optional setup team_small_planner (the student base
as orchestrator). Add a test proving an escalated step never repeats an executed
call. Run both on practice next to team_trained.
```

**Done when:** the system table shows `team_trained`, `team_escalate` and `team_small_planner` side by side.

## Stage 4: vLLM serving results (person B, in parallel)

### [ ] 4.1 Serving benchmarks

```
Using GuideLLM or vLLM's own benchmark tools (check what's installed), write
bench/ scripts that measure, on our real agent traffic replayed from traces:
(1) one vLLM server with all specialist adapters vs separate servers per merged
specialist: GPU memory and throughput; (2) a concurrency sweep (1, 8, 32
simultaneous agent sessions): throughput and p50/p95 latency; (3) prefix caching
on vs off for agent steps that share long tool schemas; (4) how often structured
outputs are valid with and without guided decoding; (5) speculative decoding on
vs off for Lightning, only if the installed vLLM supports it for this model.
Save results to runs/bench/ with the exact server flags used.
```

**Done when:** `runs/bench/summary.md` has a table for each measurement, with the server flags beside it.

## Stage 5: Final evaluation and submission

### [ ] 5.1 Final run on the test set (humans run this)

1. Run the test-set lock audit and the fairness audit from `PROMPTS.md`, and fix anything they find.
2. Freeze the code: commit and tag `final-candidate`.
3. Run `eval/decontaminate.py --against test` on all generated tasks. If it drops any, retrain the affected adapters and note it in the README.
4. Run `eval/run_eval.py --split test --final` for the specialist ladder (per API) and every system setup, 3 attempts each, with `LLM_MODE=live` and chaos off.
5. Run the Workplace Assistant and When2Call subsets on `base_seatbelt` and `single_trained`, to check that fine-tuning didn't hurt general tool use or the student's sense of when not to call a tool.
6. Run `eval.report` to build `runs/final/report.md`.

### [ ] 5.2 Publishing and write-ups

```
Read runs/final/summary.json and runs/bench/summary.md only (not data/test/ or
the traces). Write README.md: what it does, a quick start (commands to run), the
tech stack and an architecture diagram (mermaid), how to run scoring from a
fresh clone, how to reproduce the demo (env vars, and a sample .env committed as
.env.example), every model with its official parameter count, the ladder and
system results as fractions, the vLLM results, the data sources from
DATA_SOURCES.md and how each is used, honest limitations and next steps, the
team roster (names, roles, contacts), and the license (Apache-2.0, except log
lines derived from Loghub, which keep Loghub's license). Then write three short
bounty sections:
- NeMo Gym: our Gym setup, baseline vs final on the same held-out tasks and
  eval config, how to run scoring, and which change caused each improvement.
- Nemotron Lightning: what Lightning does (teacher, planner, analyst,
  escalation), why we chose it, how we optimized it, and the measured evidence
  (team_trained vs team_small_planner, escalation share, verified data yield).
- Red Hat vLLM: how vLLM fits the architecture, our configuration and
  optimizations, and the measured results.
Add scripts that publish the synthetic dataset (with its generation configs,
methodology, and a note that log lines in tool results come from Loghub under
its license) and the adapters to Hugging Face with dataset and model cards.
```

**Done when:** a teammate can run scoring from a fresh clone using only the README, and the dataset and adapters are public.

### [ ] 5.3 Demo app, video and submission

```
Build demo/app.py in Gradio: a request box, then panels showing Lightning's
plan, each specialist step, each tool call, seatbelt events, escalated steps,
and the environment state change. Show two counters: share of steps handled by
small models, and cost of the request. Give each specialist a name and an icon.
Add toggles for which specialists are active, chaos on or off, and LLM_MODE live
or replay as a fallback. Add a side-by-side run of single_base. Then write
scripts/demo_reset.sh using the "Demo safety net" prompt in PROMPTS.md.
```

**Done when:** the demo script runs twice in a row from a clean start, in live and replay mode. Then record the video following the event's Demo Video Instructions, and submit through the Airtable form with everything on the Submission Checklist.

### Demo script (about 3 minutes)

These steps are the live demo and build narration in the event's recommended video flow ("Event requirements"). Open with the team intro and pitch, close with the "so what", and keep the whole video to 3–5 minutes.

1. The problem in one sentence.
2. Send the demo request to `single_base`. It fumbles, for example by inventing a revision number or rolling back without a ticket.
3. Switch to the team with only helpdesk active. The ticket works, but Lightning's plan says no specialist can roll back.
4. Show the ladder chart for ops (base, seatbelt, r1, r2), built with NeMo Gym. Activate the ops specialist. The same request succeeds: the plan opens the ticket first, passes its ID to ops, and pods return to Running.
5. Turn on chaos and rerun. Seatbelt retries appear, and if a step escalates, the counter shows Lightning handled only that step.
6. Show the results: the system table, and one vLLM result (for example, all specialists on one server using a fraction of the memory).
7. Show one honest failure and explain it.

---

## Stretch goals (only after 5.3 is done)

- The calendar specialist, with the booking added to the demo request.
- Wrap specialists with the A2A SDK so the cards become real A2A Agent Cards.
- A chaos evaluation: score `team_trained` with `--chaos`.
- A control run for one API: the same amount of data with no failure analysis.
- An extra ladder row trained on `r1`'s data plus one outside dataset the organizers allow (for example `Salesforce/xlam-function-calling-60k`; see `data/open/DATA_SOURCES.md`), to test whether outside data helps. Keep it out of `r1` so each improvement stays isolated. It needs a new setup name in CLAUDE.md, conversion to the student's chat template, and its own row in DATA_SOURCES.md.
- Voice input with NVIDIA's Parakeet speech recognition (input only).
- Run the trained specialists offline on a laptop.
- Quantize the student with LLM Compressor, and check the adapters still load and score well.

## Cut list (if behind, cut in this order)

1. All stretch goals.
2. `team_small_planner`.
3. Round 2.
4. Benchmarks (3) and (5) in task 4.1.
5. Escalation.
6. The Workplace Assistant and When2Call regression checks.
7. Three attempts per case down to one (say so in the README).

Never cut: the locked test set, the seatbelt, the specialist ladder (it's the NeMo Gym entry), the `single_base`, `team_base` and `single_trained` comparisons, published data with its methodology, the bounty write-ups, and the demo video.
