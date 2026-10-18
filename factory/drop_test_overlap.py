"""Training examples without the ones built from tasks too close to a test case (task 5.1 step 3).

    uv run python -m factory.drop_test_overlap --api helpdesk    # and --api ops

eval/decontaminate.py --against test reports, without naming test cases, the generated tasks too close to one
(<tasks stem>.decontaminate_test.json). This writes data/generated/<api>/round_1_test_clean.jsonl: round_1.jsonl
without the examples built from those tasks (their case_id), in the same order, and round_1_test_clean.stats.json
with the counts and the report it used. factory/train_lora.py --data trains on it. Reads no test case.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from eval.run_eval import APIS, REPO, git

GENERATED = REPO / "data" / "generated"


def drop(api: str, generated: Path = GENERATED) -> dict[str, object]:
    folder = generated / api
    report_path = folder / "tasks_round_1.decontaminate_test.json"
    report = json.loads(report_path.read_text())
    too_close = {task["id"] for task in report["too_close_tasks"]}
    lines = [line for line in (folder / "round_1.jsonl").read_text().splitlines() if line.strip()]
    kept = [line for line in lines if json.loads(line)["case_id"] not in too_close]
    (folder / "round_1_test_clean.jsonl").write_text("".join(line + "\n" for line in kept))
    stats = {"script": "factory/drop_test_overlap.py", "created_at": datetime.now(timezone.utc).isoformat(),
             "git_commit": git("rev-parse", "HEAD").strip(), "from": "round_1.jsonl",
             "report": report_path.name, "tasks_too_close": len(too_close), "examples": len(lines),
             "dropped": len(lines) - len(kept), "kept": len(kept)}
    (folder / "round_1_test_clean.stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop the training examples too close to a test case.")
    parser.add_argument("--api", required=True, choices=APIS)
    args = parser.parse_args()
    stats = drop(args.api)
    print(f"{args.api}: kept {stats['kept']} of {stats['examples']} examples ({stats['dropped']} built from the "
          f"{stats['tasks_too_close']} tasks too close to a test case) in data/generated/{args.api}/round_1_test_clean.jsonl")


if __name__ == "__main__":
    main()
