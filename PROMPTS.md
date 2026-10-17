# PROMPTS.md: Prompts for executing PLAN.md in Claude Code

PLAN.md contains the build prompt for each task. This file holds the prompts that go around them: starting a session, running a task, verifying it, recovering when things break, reviewing, and the final hours. Replace anything in `<angle brackets>`.

---

## 1. First session (once)

```
Read CLAUDE.md and PLAN.md. Don't write any code. Give me:
1. A 5-line summary of what we're building and which bounties we target.
2. The hard rules in your own words.
3. Anything in either file that's unclear, contradictory, or likely to cause
   trouble, with a suggested fix.
4. The first task to do.
```

If it flags real problems, fix the files before starting Stage 0.

---

## 2. The loop for every task

Run `/clear` first, then press Shift+Tab to switch to plan mode.

**Start the task**

```
We're doing task <X.Y> from PLAN.md. Treat that task's prompt as your
instructions, and read any CLAUDE.md sections it depends on. Propose a plan:
the files you'll create or change, in what order, how you'll test as you go,
and any questions for me. Keep it under 15 lines.
```

Read the plan. Push back on anything that adds scope or touches `data/test/`. Then:

```
Approved. Implement it. If you hit something the plan didn't cover, stop and
ask instead of improvising.
```

**Verify**

```
Run the "Done when" check for task <X.Y> exactly as written in PLAN.md and show
me the raw output. Don't fix anything yet. Tell me plainly: passed or failed?
```

**Wrap up** (only after it passed)

```
Task <X.Y> passed. Tick its box in PLAN.md, update the Commands section of
CLAUDE.md if anything changed, then commit with a message that says what was
built and any known issues.
```

**Resume** (new session, or after `/clear` mid-task)

```
Read CLAUDE.md, PLAN.md, and the output of `git log --oneline -15` and
`git status`. Tell me which tasks are done, what's in progress, and the next
step. Don't change anything.
```

---

## 3. When things break

**General debugging**

```
This isn't working:
<paste error or unexpected output>
Before changing any code, list the 3 most likely causes, ranked, with a cheap
way to test each. Then test the most likely one and show me the result.
```

**Going in circles** (use when the same error comes back twice)

```
Stop. Summarize what you've tried and why each attempt failed. Then propose one
genuinely different approach. Don't edit code until I approve it.
```

**Unsure about a library**

```
Before writing more code against <NeMo Gym / vLLM / TRL / Unsloth>, check the
installed version, then read its docs or source for <feature>. Show me the exact
functions, config keys or CLI commands you'll use and where you found them.
```

**Stuck on NeMo Gym**

```
We're stuck getting <describe> to work in NeMo Gym. Find the closest working
example in the installed package or its docs (for example the Workplace
Assistant environment, which also uses state-matching verification), run it
unchanged to confirm the setup works, then show me the smallest diff between it
and what we need.
```

**Tool calls don't parse after training** (smoke check below 19/20)

```
The smoke check on <adapter> parsed only <N>/20. Put three things side by side:
a raw completion from the adapter, the "text" field of one training example,
and the format vLLM's <parser> parser expects. Find the mismatch (chat template,
special tokens, tool-call wrapper) and explain it before fixing anything.
```

**The filter keeps too few rollouts**

```
Only <K> of <N> Lightning rollouts passed the verifier. Split the failures by
origin (sampled vs written) and by reason, show 3 examples of the most common
reason, and tell me whether the problem is the sampler, the writer, Lightning,
the environment, or the verifier.
```

**Out of GPU memory**

```
We ran out of GPU memory:
<paste error>
Show current usage with nvidia-smi and the memory settings in configs/gpu.md.
List what we could change (GPU memory fraction per server, max model length,
max LoRA rank, batch size, pausing a server during training, Lightning's
NVFP4 build if the GPU supports it) with the expected effect of each, and
recommend the smallest change.
```

**A factory round didn't help**

```
Round <N> didn't beat round <N-1> on practice (<scores>). Without touching
data/test/, check: (1) the failure clusters before and after, (2) how many
targeted examples each cluster got, and whether 3 random ones actually follow
the cluster's guidance, and (3) whether the training loss looks normal. Tell me
the most likely reason and the cheapest fix.
```

**The seatbelt is doing too much work**

```
On practice, the seatbelt retried on <N> of <M> steps. Group the retries by
cause (unknown tool, schema error, environment error, self-check said
incomplete), show 2 examples of the biggest group, and tell me whether the fix
belongs in the prompts, the tool descriptions, the environment's error
messages, or the training data.
```

**Too many escalations**

```
team_escalate escalated <N> of <M> steps on practice. Show which specialists
and which kinds of request escalate most, with 3 examples, and tell me whether
those steps would be better fixed by another factory round for that specialist.
```

**Most team failures are routing**

```
Most team failures on practice are labeled routing. Show 5 failed plans next to
the APIs in those cases' gold_calls. Suggest changes to the orchestrator prompt,
the plan schema, or the specialist card descriptions.
```

**The environment behaves inconsistently**

```
The environment gave different results for the same calls: <describe>. Find
where state or randomness leaks (unseeded IDs, real time, state shared between
tasks, chaos left on) and add a determinism test that would have caught it
before fixing it.
```

**Benchmark numbers look wrong**

```
The benchmark result for <measurement> looks off: <numbers>. Check that the
server flags match what runs/bench/ recorded, that the GPU wasn't shared with
another job during the run, that warm-up requests were excluded, and that the
replayed traffic matches our real agent requests. Rerun once and compare.
```

---

## 4. Reviews and integration

**Before merging any branch**

```
Review this branch's diff against main as a strict reviewer. Check the hard
rules in CLAUDE.md (especially anything touching data/test/, rule 2 on model
sizes, and rule 7 on equal conditions), the data formats in common/schemas.py,
determinism of the environment, and anything that would break the demo. List
problems by severity. Don't fix anything yet.
```

**Merging a teammate's work**

```
Merge branch <name> into main. Run pytest and the smoke checks. If anything
conflicts with the data formats in CLAUDE.md, stop and show me the conflict
instead of resolving it yourself.
```

**Test-set lock audit** (run before 5.1)

```
Search the code (not the data) for every place that reads from data/test/.
Anything outside eval/run_eval.py with --final and eval/decontaminate.py with
--against test is a bug. Show me every hit.
```

**Fairness audit** (run before 5.1)

```
Without opening data/test/, check that every setup uses the same NeMo Gym eval
config, the same tools per API, the same seatbelt settings (except base_raw),
the same max steps, the same temperature and the same attempts per case.
Confirm that only team_escalate can escalate, that chaos is off and LLM_MODE is
live for eval runs, and that every setup is priced with the same formula. Show
me any differences.
```

**Rules audit** (run before 5.1)

```
List every generative model the system uses anywhere (agents, orchestrator,
teacher, analyst, self-check, any speculative-decoding draft model) with its
official total parameter count from configs/models.yaml, and check them against
the 33B rule as the organizers clarified it in PLAN.md task 0.1. Then check that
every dataset in data/generated/ has its generation config saved, and that
data/open/DATA_SOURCES.md lists every outside source with a license.
```

---

## 5. Time checks

Every few hours:

```
It's <time>. The feature freeze is at <time> and the deadline is at <time>.
Compare progress with the time budget in PLAN.md. Which tasks are behind? Using
the cut list, tell me what to cut now so Stage 5 still gets its full time.
```

---

## 6. Final hours

Task 5.1 (the test-set run) is done by humans with the steps in PLAN.md, not by a prompt.

**README and write-up numbers check**

```
Compare every number in README.md and the three bounty sections with
runs/final/report.md, runs/final/summary.json and runs/bench/summary.md. List
every mismatch. Don't open data/test/ or the final traces.
```

**Fresh-clone check** (the NeMo Gym bounty asks how to run scoring)

```
Clone the repo into a new folder and follow only the README to run scoring on
the practice split for base_seatbelt and r1. Tell me every step where the README
was wrong, missing, or assumed something that isn't in the repo.
```

**Demo safety net**

```
Write scripts/demo_reset.sh: turn chaos off, reset the environment to the
demo's incident fixture, activate only the helpdesk specialist in
agents/registry.json, and send one warm-up request to Lightning and the
student server. Check that replay mode has recorded responses for every step of
the demo script. Then walk through the demo script in PLAN.md and tell me which
step is most likely to fail live and what our fallback is.
```

**Judge Q&A prep**

```
Based on the code, PLAN.md, runs/final/report.md and runs/bench/summary.md,
list the 8 toughest questions an NVIDIA or Red Hat judge might ask, including
"why not just use Lightning for everything?", and draft an honest 2-sentence
answer for each, including where our results are weak.
```
