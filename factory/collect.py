"""Collects Lightning rollouts on generated tasks through NeMo Gym.

    uv run python -m factory.collect --tasks data/generated/helpdesk/pilot/tasks_pilot.jsonl --run-id pilot_helpdesk

Lightning's vLLM server must be running (scripts/serve_lightning.sh) and port 11000 free. Like
eval/run_eval.py, this starts the Gym servers for the `lightning` setup (gym_configs/lightning.yaml: the
itops agent with the seatbelt), collects the rollouts with `gym eval run`, and stops the servers. The
agent's settings are configs/eval.yaml with the `collect` fields of configs/factory.yaml changed
(thinking on, one attempt per task), so the tools, seatbelt and max steps are the eval's. This isn't an
eval: factory/filter.py turns the rollouts into training data.

Writes to runs/<run_id>/:
    config.json         arguments, the factory config, the resolved agent config, the task files, the git commit
    git_diff.patch      uncommitted changes, when the tree is dirty
    agent_config.yaml   the agent's settings (configs/eval.yaml + the collect fields)
    tasks.jsonl         the Gym rows (gym_env/tasks.py)
    rollouts.jsonl      Gym's rollouts, next to Gym's side files, gym_env.log and gym_eval.log
Refuses tasks under data/test/ (hard rule 5: the factory never sees test data) and runs with chaos on.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from common.eval_config import EVAL_CONFIG, EvalConfig
from eval.run_eval import REPO, collect, git_commit, server_chaos_enabled, start_servers, stop_servers
from factory.sampler import FACTORY_CONFIG, load_factory_config
from gym_env.tasks import case_to_row, read_cases

SETUP = "lightning"
TEST_DIR = REPO / "data" / "test"


def agent_config(factory_config: dict[str, Any], eval_path: Path = EVAL_CONFIG) -> dict[str, Any]:
    """configs/eval.yaml with the factory's collect fields changed, validated as an EvalConfig."""
    settings = yaml.safe_load(eval_path.read_text()) | factory_config["collect"]
    EvalConfig.model_validate(settings)
    return settings


def refusal(tasks: Path, settings: dict[str, Any], server_chaos: bool) -> str | None:
    if tasks.resolve().is_relative_to(TEST_DIR):
        return "data/test/ is locked (hard rules 4 and 5): the factory never sees test data."
    if settings["chaos"] or server_chaos:
        return "chaos is on; rollouts for training are collected without it."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Lightning rollouts on generated tasks through NeMo Gym.")
    parser.add_argument("--tasks", type=Path, required=True, help="a tasks_<name>.jsonl file from factory/writer.py")
    parser.add_argument("--run-id", help="default: <time>_collect_<tasks name>")
    parser.add_argument("--config", type=Path, default=FACTORY_CONFIG, help="factory config")
    args = parser.parse_args()

    factory_config = load_factory_config(args.config)
    settings = agent_config(factory_config)
    if reason := refusal(args.tasks, settings, server_chaos_enabled()):
        raise SystemExit(f"Refusing to run: {reason}")
    cases = read_cases(args.tasks)
    if not cases:
        raise SystemExit(f"No tasks in {args.tasks}.")

    started = datetime.now(timezone.utc)
    run_id = args.run_id or f"{started:%Y%m%d-%H%M%S}_collect_{args.tasks.stem.removeprefix('tasks_')}"
    run_dir = REPO / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    agent_path = run_dir / "agent_config.yaml"
    agent_path.write_text(yaml.safe_dump(settings, sort_keys=False))
    tasks_path = args.tasks.resolve()
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id,
        "script": "factory/collect.py",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "started_at": started.isoformat(),
        "setup": SETUP,
        "tasks": str(tasks_path.relative_to(REPO)) if tasks_path.is_relative_to(REPO) else str(tasks_path),
        "task_count": len(cases),
        "factory_config": factory_config,
        "agent_config": settings,
        **git_commit(run_dir),
    }, indent=2))
    rows = run_dir / "tasks.jsonl"
    rows.write_text("".join(json.dumps(case_to_row(case)) + "\n" for case in cases))

    attempts = settings["attempts_per_case"]
    print(f"{run_id}: {len(cases)} tasks x {attempts} attempt(s) with {SETUP}; starting Gym servers")
    servers = start_servers(SETUP, agent_path, False, run_dir / "gym_env.log")
    try:
        returncode = collect(rows, run_dir / "rollouts.jsonl", attempts, run_dir / "gym_eval.log")
    finally:
        stop_servers(servers)

    rollouts = run_dir / "rollouts.jsonl"
    collected = [json.loads(line) for line in rollouts.read_text().splitlines() if line.strip()] if rollouts.exists() else []
    expected = len(cases) * attempts
    passed = sum(row.get("reward") == 1.0 for row in collected)
    print(f"Collected {len(collected)}/{expected} rollouts; reward 1 on {passed}/{len(collected)} "
          "(before the filter's no-failed-calls check)")
    if returncode != 0 or len(collected) != expected:
        print(f"PARTIAL RUN: gym eval run exited {returncode}. See {run_dir / 'gym_eval.log'}.")
    print(f"Results in {run_dir}\nNext: python -m factory.filter --run {run_id} ...")


if __name__ == "__main__":
    main()
