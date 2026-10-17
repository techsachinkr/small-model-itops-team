"""Converts our cases to NeMo Gym's task format.

This is the only place that knows Gym's row format. A row is
`{task_id, case, responses_create_params: {input, tools}}`: Gym's agent sends the whole row to
/seed_session and /verify, which read `case`; the model sees only `responses_create_params`. A collected
rollout adds `response`, whose final answer rollout_answer() reads.

    uv run python -m gym_env.tasks --cases <cases.jsonl> --output <rows.jsonl>
"""

import argparse
import json
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from common.schemas import Case
from gym_env.env import tool_schemas
from gym_env.state import load_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = REPO_ROOT / "data" / "test"

SYSTEM_PROMPT = """You are an IT operations assistant. Use the tools to carry out the user's request. If information you need is missing, ask the user instead of guessing.
Current user: {user_id}
Current time: {now}"""


def system_prompt(case: Case) -> str:
    now = datetime.fromisoformat(load_fixture(case.fixture).now.replace("Z", "+00:00"))
    return SYSTEM_PROMPT.format(user_id=case.user_id, now=now.strftime("%A %Y-%m-%d %H:%M UTC"))


def case_to_row(case: Case, apis: Iterable[str] | None = None) -> dict[str, Any]:
    """One Gym task row. Tools are those of the case's APIs unless `apis` overrides them."""
    return {
        "task_id": case.id,
        "case": case.model_dump(mode="json"),
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system_prompt(case)},
                {"role": "user", "content": case.request},
            ],
            "tools": tool_schemas(apis or case.apis),
        },
    }


def rollout_answer(row: dict[str, Any]) -> str:
    """A collected rollout's final answer: the last assistant message's text, as gym_env/server.py's
    final_answer() takes it from the response it verifies."""
    answer = ""
    for item in row["response"]["output"]:
        if item.get("type") == "message" and item.get("role") == "assistant":
            answer = "\n".join(part["text"] for part in item["content"] if part.get("type") == "output_text")
    return answer


def read_cases(path: Path) -> list[Case]:
    return [Case.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cases (our schema) to NeMo Gym task rows.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cases.resolve().is_relative_to(TEST_DIR):
        raise SystemExit("data/test/ is locked (CLAUDE.md rule 4); only eval/run_eval.py --final reads it.")
    rows = [case_to_row(case) for case in read_cases(args.cases)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(f"{len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
