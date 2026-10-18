"""Runs tool calls on a fresh copy of a fixture and prints what each returns, for working out a case's gold calls and
answer strings (TEST_SET_GUIDE.md). The calls run one after another on the same copy, as gold calls do, so a ticket
the first call opens can be used by the next. Changes nothing on disk.

    .venv/bin/python scripts/checks/try_calls.py default list_devices '{"employee_id": "E1003"}'
    .venv/bin/python scripts/checks/try_calls.py incident_prod_payments rollout_history '{"namespace": "prod", "deployment": "payments"}'

Its output shows what your case does: keep it to yourself (Claude must not see test cases).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gym_env.env import Env  # noqa: E402


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 3 or len(args) % 2 == 0:
        raise SystemExit(__doc__)
    env = Env(args[0])
    for tool, raw in zip(args[1::2], args[2::2]):
        result = env.call(tool, json.loads(raw))
        print(f"{tool} {raw} -> {result.status}\n{json.dumps(result.output, indent=1)}\n")


if __name__ == "__main__":
    main()
