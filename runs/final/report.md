| setup | api | split | cases passed | attempts | mean latency (s) | $ per 1,000 requests | escalated steps | small-model steps | thinking | run |
|---|---|---|---|---|---|---|---|---|---|---|
| base_raw | helpdesk | test | 6/10 | 30/30 | 2.7 | 0.14 | - | - | off | final/base_raw_helpdesk |
| base_seatbelt | helpdesk | test | 5/10 | 30/30 | 4.0 | 0.17 | - | - | off | final/base_seatbelt_helpdesk |
| r1 | helpdesk | test | 8/10 | 30/30 | 4.2 | 0.15 | - | - | off | final/r1_helpdesk |
| lightning | helpdesk | test | 6/10 | 30/30 | 4.8 | 0.27 | - | - | off | final/lightning_helpdesk |
| base_raw | ops | test | 5/10 | 30/30 | 3.1 | 0.18 | - | - | off | final/base_raw_ops |
| base_seatbelt | ops | test | 5/10 | 30/30 | 4.3 | 0.23 | - | - | off | final/base_seatbelt_ops |
| r1 | ops | test | 7/10 | 30/30 | 5.4 | 0.23 | - | - | off | final/r1_ops |
| lightning | ops | test | 9/10 | 30/30 | 6.0 | 0.39 | - | - | off | final/lightning_ops |
| single_base | all | test | 7/15 | 45/45 | 12.5 | 0.54 | - | - | off | final/single_base |
| single_trained | all | test | 8/15 | 45/45 | 12.6 | 0.42 | - | - | off | final/single_trained |
| single_lightning | all | test | 10/15 | 45/45 | 11.7 | 0.79 | - | - | off | final/single_lightning |
| team_base | all | test | 6/15 | 45/45 | 16.4 | 0.58 | - | - | off | final/team_base |
| team_trained | all | test | 8/15 | 45/45 | 21.5 | 0.53 | - | - | off | final/team_trained |
| team_escalate | all | test | 7/15 | 45/45 | 20.7 | 0.53 | 0 | 87/87 | off | final/team_escalate |
| team_small_planner | all | test | 5/15 | 45/45 | 9.4 | 0.38 | - | - | off | final/team_small_planner |

Cost: GPU time at full load, at configs/eval.yaml's prices per 1M tokens (one H200 at $6.53 an hour over each model's measured throughput). Adapters are priced as the student base model, which understates the setups that run them (r1, single_trained, team_trained, team_escalate): LoRA serving is slower, 46% more time per output token at 1 session in task 4.1 (runs/bench/summary.md).

Attempts passed, of 3 per case (a case passes with 2 of 3): helpdesk ladder base_raw 17/30, base_seatbelt 15/30, r1 23/30,
lightning 18/30; ops ladder 18/30, 18/30, 21/30, 27/30; system single_base 20/45, single_trained 24/45, single_lightning
30/45, team_base 18/45, team_trained 25/45, team_escalate 24/45, team_small_planner 17/45 (each run's per_case.json).

Notes:
- The test set (data/test/, 35 cases: 10 helpdesk, 10 ops, 15 multi-API) is AI-written. No human-written set existed, so
  the user changed CLAUDE.md hard rule 4: a fresh Claude Code subagent wrote it from TEST_SET_GUIDE.md, the fixtures and
  the tools, told not to read the practice cases, training data, prompts or results; the session that built the system
  never opened it, and nothing was tuned on it.
- Before the run, eval/decontaminate.py --against test found 54 of 1,085 helpdesk and 203 of 1,016 ops training examples
  with the same state changes as a test case. helpdesk_r1, ops_r1 and all_r1 were retrained without them
  (data/generated/<api>/round_1_test_clean.jsonl) and are what these runs used; the practice results came from the
  original adapters.
- All 15 runs are commit 997e6a0 (tag final-candidate), split test, chaos off, LLM_MODE live, 3 attempts per case, every
  attempt collected. Every run after the first is marked dirty only because of the earlier final runs' own files in
  runs/final/ (their git_diff.patch files touch nothing else; they are kept out of git).
- team_escalate escalated no step: none of its 87 steps was given up on.
- With 10 or 15 cases, a difference of one case is within noise; the attempts are a finer measure.
