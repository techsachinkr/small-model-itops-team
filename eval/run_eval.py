"""Runs one setup on a split through NeMo Gym and writes the results to runs/<run_id>/.

    uv run python -m eval.run_eval --setup base_seatbelt --api helpdesk --split practice

The vLLM server for the setup's model must already be running (scripts/serve_*.sh). This script
starts the Gym servers for the setup (resources_servers/itops/configs/itops.yaml, gym_configs/eval.yaml
and gym_configs/<setup>.yaml; adapter setups like r1 run the API's own adapter, "<api>_r1"), collects
`attempts_per_case` rollouts per case with `gym eval run`,
stops the servers, and writes to runs/<run_id>/:
    config.json     arguments, the resolved eval config, the Gym config files, the git commit
    git_diff.patch  uncommitted changes, when the tree is dirty (commit + patch = the code that ran)
    tasks.jsonl     the Gym rows (gym_env/tasks.py)
    rollouts.jsonl  Gym's rollouts, next to Gym's own side files and logs
    traces.jsonl    one Trace per attempt (common/schemas.py)
    per_case.json   passed attempts per case
    summary.json    the RunSummary: passes as "k/n", mean latency, cost per 1,000 requests

A case passes if at least `pass_min_attempts` of its attempts pass; an attempt Gym failed to collect
counts as failed, and the run is reported as partial. Refuses to start when LLM_MODE=replay, when
chaos is on without --chaos, or for --split test without --final (CLAUDE.md "Scoring", hard rule 4).
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from common.eval_config import EVAL_CONFIG, REPO, EvalConfig, load_eval_config
from common.schemas import Case, RunSummary, SetupSummary, Step, Trace
from gym_env.tasks import case_to_row, read_cases

GYM = Path(sys.executable).parent / "gym"
GYM_CONFIGS = REPO / "gym_configs"
ITOPS_CONFIG = REPO / "resources_servers" / "itops" / "configs" / "itops.yaml"
SETUPS = sorted(path.stem for path in GYM_CONFIGS.glob("*.yaml") if path.stem != "eval")
ADAPTER_SETUPS = ("r1", "r2")  # one adapter per API; their gym_configs leave the model unset
APIS = ("helpdesk", "ops")
HEAD_SERVER_PORT = 11000  # Gym's default
READY = re.compile(r"All (\d+) / \1 servers ready")
SERVER_TIMEOUT_S = 900  # the first start builds each server's venv


def refusal(args: argparse.Namespace, environ: Mapping[str, str], config: EvalConfig, server_chaos: bool) -> str | None:
    """Why this run must not start, or None."""
    if environ.get("LLM_MODE", "live") == "replay":
        return "LLM_MODE=replay: evals never run on recorded responses (CLAUDE.md \"Scoring\")."
    if args.split == "test" and not args.final:
        return "data/test/ is locked (hard rule 4): --split test runs only with --final, when a human asks for it."
    if (config.chaos or server_chaos) and not args.chaos:
        return f"Chaos is on in {EVAL_CONFIG.name} or {ITOPS_CONFIG.name}; pass --chaos to run with it."
    return None


def server_chaos_enabled() -> bool:
    return bool(yaml.safe_load(ITOPS_CONFIG.read_text())["itops"]["resources_servers"]["itops"]["chaos"]["enabled"])


def load_cases(split: str, api: str) -> list[Case]:
    """The split's single-API cases for `api` (the specialist ladder runs on these)."""
    path = REPO / "data" / split / f"{api}.jsonl"
    cases = [case for case in read_cases(path) if case.apis == [api]]
    if not cases:
        raise SystemExit(f"No {api} cases in {path}.")
    return cases


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def git_commit(run_dir: Path) -> dict[str, Any]:
    """The commit, and the uncommitted changes to tracked files saved as git_diff.patch."""
    dirty = bool(git("status", "--porcelain").strip())
    if dirty:
        (run_dir / "git_diff.patch").write_text(git("diff", "HEAD"))
    return {"git_commit": git("rev-parse", "HEAD").strip(), "git_dirty": dirty}


def gym_configs(setup: str) -> list[Path]:
    return [ITOPS_CONFIG, GYM_CONFIGS / "eval.yaml", GYM_CONFIGS / f"{setup}.yaml"]


def adapter_model(setup: str, api: str) -> str | None:
    """The model an adapter setup runs: the API's own adapter, which scripts/serve_student.sh serves under its
    folder's name (adapters/helpdesk_r1 -> "helpdesk_r1"). None for setups whose file names the model."""
    return f"{api}_{setup}" if setup in ADAPTER_SETUPS else None


def start_servers(setup: str, config_path: Path, chaos: bool, log_path: Path, model: str | None = None) -> subprocess.Popen:
    """`model` replaces the setup's policy model (an adapter setup's file leaves it unset)."""
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", HEAD_SERVER_PORT)) == 0:
            raise SystemExit(f"Something is already listening on port {HEAD_SERVER_PORT} (a Gym head server?). Stop it first.")
    command = [str(GYM), "env", "start"]
    for path in gym_configs(setup):
        command += ["--config", str(path.relative_to(REPO))]
    command += [
        "+uv_venv_dir=runs/gym_venvs",
        f"++itops_agent.responses_api_agents.itops_agent.eval_config={config_path.relative_to(REPO)}",
    ]
    if model:
        command.append(f"++policy_model_name={model}")
    if chaos:
        command.append("++itops.resources_servers.itops.chaos.enabled=true")
    # Gym builds a venv per server; keep their dependencies within hard rule 1 too. Ray's uv hook stays off: under
    # `uv run`, every server's ray.init() would set Ray's workers up to run through uv, and stop because
    # pyproject.toml isn't in the server's folder (task 2.2). Off, the servers start as under .venv/bin/python,
    # as in the baselines.
    exclude_newer = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["uv"]["exclude-newer"]
    env = os.environ | {"UV_EXCLUDE_NEWER": exclude_newer, "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0"}
    log = log_path.open("w")
    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + SERVER_TIMEOUT_S
    while not READY.search(log_path.read_text()):
        if process.poll() is not None or time.monotonic() > deadline:
            stop_servers(process)
            tail = "\n".join(log_path.read_text().splitlines()[-20:])
            raise SystemExit(f"Gym servers didn't start; see {log_path}:\n{tail}")
        time.sleep(2)
    return process


def stop_servers(process: subprocess.Popen) -> None:
    """Ctrl+C for the whole process group, as when `gym env start` runs in a terminal."""
    for sig, wait_s in ((signal.SIGINT, 60), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(wait_s)
        except subprocess.TimeoutExpired:
            continue


def collect(tasks: Path, rollouts: Path, attempts: int, log_path: Path) -> int:
    command = [str(GYM), "eval", "run", "--no-serve", "--agent", "itops_agent", "--input", str(tasks),
               "--output", str(rollouts), "--num-repeats", str(attempts)]
    with log_path.open("w") as log:
        return subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode


def to_trace(row: dict[str, Any], run_id: str, setup: str, config: EvalConfig) -> Trace:
    trace = row["trace"]
    steps = [Step.model_validate(step) for step in trace["steps"]]
    prices = config.prices_for(steps[0].model)
    cost = (trace["prompt_tokens"] * prices.prompt + trace["completion_tokens"] * prices.completion) / 1e6
    return Trace(
        run_id=run_id,
        setup=setup,
        case_id=row["task_id"],
        attempt=row["_ng_rollout_index"] + 1,
        steps=steps,
        prompt_tokens=trace["prompt_tokens"],
        completion_tokens=trace["completion_tokens"],
        latency_s=trace["latency_s"],
        cost_usd=cost,
        passed=row["reward"] == 1.0,
    )


def summarize(
    traces: list[Trace], cases: list[Case], setup: str, config: EvalConfig
) -> tuple[RunSummary, dict[str, dict[str, Any]]]:
    by_case: dict[str, list[Trace]] = defaultdict(list)
    for trace in traces:
        by_case[trace.case_id].append(trace)
    per_case = {}
    for case in cases:
        attempts = by_case.get(case.id, [])
        passed = sum(trace.passed for trace in attempts)
        per_case[case.id] = {
            "kind": case.kind,
            "passed_attempts": f"{passed}/{config.attempts_per_case}",
            "collected_attempts": len(attempts),
            "passed": passed >= config.pass_min_attempts,
        }
    passes = sum(result["passed"] for result in per_case.values())
    summary = SetupSummary(
        passes=f"{passes}/{len(cases)}",
        mean_latency_s=sum(t.latency_s for t in traces) / len(traces) if traces else 0.0,
        cost_per_1k_usd=1000 * sum(t.cost_usd for t in traces) / len(traces) if traces else 0.0,
    )
    return RunSummary({setup: summary}), per_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one setup on a split through NeMo Gym.")
    parser.add_argument("--setup", required=True, choices=SETUPS)
    parser.add_argument("--api", required=True, choices=APIS)
    parser.add_argument("--split", required=True, choices=["practice", "test"])
    parser.add_argument("--final", action="store_true", help="required for --split test")
    parser.add_argument("--chaos", action="store_true", help="run with chaos on (demo only)")
    parser.add_argument("--config", type=Path, default=EVAL_CONFIG, help="eval config (default configs/eval.yaml)")
    parser.add_argument("--run-id", help="default: <time>_<setup>_<api>_<split>")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_eval_config(config_path)
    if reason := refusal(args, os.environ, config, server_chaos_enabled()):
        raise SystemExit(f"Refusing to run: {reason}")

    started = datetime.now(timezone.utc)
    run_id = args.run_id or f"{started:%Y%m%d-%H%M%S}_{args.setup}_{args.api}_{args.split}"
    run_dir = REPO / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    cases = load_cases(args.split, args.api)
    adapter = adapter_model(args.setup, args.api)
    (run_dir / "config.json").write_text(json.dumps({
        "run_id": run_id,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "started_at": started.isoformat(),
        "eval_config": config.model_dump(),
        "gym_configs": [str(path.relative_to(REPO)) for path in gym_configs(args.setup)],
        "adapter": adapter,
        "cases": len(cases),
        **git_commit(run_dir),
    }, indent=2))
    tasks = run_dir / "tasks.jsonl"
    tasks.write_text("".join(json.dumps(case_to_row(case)) + "\n" for case in cases))

    print(f"{run_id}: {len(cases)} cases x {config.attempts_per_case} attempts; starting Gym servers")
    servers = start_servers(args.setup, config_path, args.chaos, run_dir / "gym_env.log", adapter)
    try:
        returncode = collect(tasks, run_dir / "rollouts.jsonl", config.attempts_per_case, run_dir / "gym_eval.log")
    finally:
        stop_servers(servers)

    rollouts = run_dir / "rollouts.jsonl"
    rows = [json.loads(line) for line in rollouts.read_text().splitlines() if line.strip()] if rollouts.exists() else []
    traces = [to_trace(row, run_id, args.setup, config) for row in rows]
    (run_dir / "traces.jsonl").write_text("".join(trace.model_dump_json() + "\n" for trace in traces))
    summary, per_case = summarize(traces, cases, args.setup, config)
    (run_dir / "per_case.json").write_text(json.dumps(per_case, indent=2))
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2))

    expected = len(cases) * config.attempts_per_case
    for case_id, result in per_case.items():
        print(f"  {case_id:<12} {result['kind']:<12} {result['passed_attempts']}  {'pass' if result['passed'] else 'FAIL'}")
    result = summary.root[args.setup]
    print(f"{args.setup} on {args.api} {args.split}: {result.passes} cases passed; "
          f"mean latency {result.mean_latency_s:.1f} s; ${result.cost_per_1k_usd:.2f} per 1,000 requests")
    if returncode != 0 or len(traces) != expected:
        print(f"PARTIAL RUN: gym eval run exited {returncode}; {len(traces)}/{expected} attempts collected "
              f"(missing attempts count as failed). See {run_dir / 'gym_eval.log'}.")
    print(f"Results in {run_dir}")


if __name__ == "__main__":
    main()
