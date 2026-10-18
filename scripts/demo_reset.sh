#!/usr/bin/env bash
# The demo's safety net (PROMPTS.md "Demo safety net", task 5.3). Run it just before presenting:
#   scripts/demo_reset.sh
# - Chaos off in configs/eval.yaml and the itops resources server's config (the demo turns chaos on per run, in its
#   own environment).
# - Nothing to reset in the environment: every demo run starts from a fresh copy of the incident fixture.
# - agents/registry.json keeps every specialist active, because the evals refuse otherwise; the demo page starts with
#   only helpdesk ticked, as the demo script does (the user's decision, 2026-10-18).
# - Warm-up: one short request to Lightning and to the student and each specialist adapter, so the first live run
#   doesn't pay for loading.
# - Replay: demo/rehearse.py --mode replay must bring back every step of the demo script as recorded, the fallback if
#   a server fails on stage.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO" || exit 1
PY=$REPO/.venv/bin/python
status=0

"$PY" - <<'EOF' || status=1
from agents.specialist import inactive_names
from common.eval_config import load_eval_config
from eval.run_eval import server_chaos_enabled

problems = []
if load_eval_config().chaos or server_chaos_enabled():
    problems.append("chaos is on in configs/eval.yaml or resources_servers/itops/configs/itops.yaml")
if inactive_names():
    problems.append(f"agents/registry.json has specialists switched off: {inactive_names()}")
print("\n".join(f"PROBLEM: {p}" for p in problems) or "chaos off; every specialist active in the registry")
raise SystemExit(1 if problems else 0)
EOF

for target in "8000 lightning" "8001 student" "8001 helpdesk_r1" "8001 ops_r1"; do
  read -r port model <<<"$target"
  if curl -sf --max-time 120 "localhost:$port/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\": \"$model\", \"prompt\": \"Hello\", \"max_tokens\": 1}" >/dev/null; then
    echo "warm: $model on port $port"
  else
    echo "NOT ANSWERING: $model on port $port (live runs will fail; replay still works)"
    status=1
  fi
done

"$PY" -m demo.rehearse --mode replay || status=1
if ((status)); then echo "demo_reset: problems above"; else echo "demo_reset: ready"; fi
exit $status
