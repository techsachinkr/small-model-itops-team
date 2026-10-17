"""Check (task 2.1): re-score stored Gym rollouts after the verifier's answer check began reading look-alike
hyphens and spaces as ASCII. Only the answer check changed, so each attempt keeps its recorded state match
and seatbelt outcome, and its final answer is checked again with gym_env.verify.answer_check. Reads only
runs/ (never data/test/); writes nothing.

    .venv/bin/python scripts/checks/rescore_answers.py runs/baseline_* runs/round1_helpdesk runs/round1_ops
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.eval_config import load_eval_config  # noqa: E402
from common.schemas import Case  # noqa: E402
from gym_env.verify import answer_check  # noqa: E402


def final_answer(row: dict) -> str:
    """The last assistant message's text, as gym_env/server.py's final_answer() takes it."""
    answer = ""
    for item in row["response"]["output"]:
        if item.get("type") == "message" and item.get("role") == "assistant":
            answer = "\n".join(part["text"] for part in item["content"] if part.get("type") == "output_text")
    return answer


def passed(row: dict, answer_ok: bool) -> bool:
    gave_up = any(e["kind"] == "gave_up" for step in row["trace"]["steps"] for e in step["seatbelt_events"])
    return bool(row.get("state_match")) and answer_ok and not gave_up


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    need = load_eval_config().pass_min_attempts
    for run in args.runs:
        rows = [json.loads(line) for line in (run / "rollouts.jsonl").read_text().splitlines() if line.strip()]
        before, after = defaultdict(int), defaultdict(int)
        flipped = []
        for row in rows:
            missing, forbidden = answer_check(Case.model_validate(row["case"]), final_answer(row))
            old, new = row["reward"] == 1.0, passed(row, not missing and not forbidden)
            before[row["task_id"]] += old
            after[row["task_id"]] += new
            if old != new:
                flipped.append(f"{row['task_id']}#{row['_ng_rollout_index'] + 1}: {'fail -> pass' if new else 'pass -> fail'}")
        line = f"{run.name}: attempts passed {sum(before.values())} -> {sum(after.values())} of {len(rows)}"
        if len(rows) > len(before):  # an eval run, with several attempts per case
            line += (f"; cases passed (>= {need} attempts) {sum(n >= need for n in before.values())} -> "
                     f"{sum(n >= need for n in after.values())} of {len(before)}")
        print(line)
        for line in flipped[:10]:
            print(f"  {line}")
        if len(flipped) > 10:
            print(f"  ... and {len(flipped) - 10} more")


if __name__ == "__main__":
    main()
