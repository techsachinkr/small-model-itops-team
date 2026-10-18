"""Builds result tables and reports from run summaries.

    uv run python -m eval.report --run <run_id> [--run <run_id> ...] [--output <file.md>]

One row per run, from runs/<run_id>/config.json and summary.json (written by eval/run_eval.py): setup, API,
split, cases passed as "k/n", attempts collected, mean latency, cost per 1,000 requests, and for team_escalate the
steps escalated to Lightning and the share of steps small models handled ("k/n" of the steps run; "-" for every
other setup, which never escalates). A run that collected fewer attempts than it should is marked partial (hard
rule 8).
"""

import argparse
import json
from pathlib import Path

from common.eval_config import REPO
from common.schemas import RunSummary, SetupSummary

RUNS = REPO / "runs"


def escalation_columns(summary: SetupSummary) -> list[str]:
    """Steps escalated to Lightning, and steps small models handled out of the steps run (team_escalate only)."""
    if summary.small_model_steps is None:
        return ["-", "-"]
    small, steps = (int(part) for part in summary.small_model_steps.split("/"))
    return [str(steps - small), summary.small_model_steps]


def row(run_id: str) -> list[str]:
    run_dir = RUNS / run_id
    config = json.loads((run_dir / "config.json").read_text())
    args, eval_config = config["args"], config["eval_config"]
    (setup, summary), = RunSummary.model_validate_json((run_dir / "summary.json").read_text()).root.items()
    expected = config["cases"] * eval_config["attempts_per_case"]
    collected = sum(1 for line in (run_dir / "traces.jsonl").read_text().splitlines() if line.strip())
    status = "" if collected == expected else " (partial)"
    return [
        setup,
        args.get("api") or "all",
        args["split"],
        f"{summary.passes}{status}",
        f"{collected}/{expected}",
        f"{summary.mean_latency_s:.1f}",
        f"{summary.cost_per_1k_usd:.2f}",
        *escalation_columns(summary),
        "on" if eval_config.get("thinking", True) else "off",
        run_id,
    ]


def table(run_ids: list[str]) -> str:
    header = ["setup", "api", "split", "cases passed", "attempts", "mean latency (s)", "$ per 1,000 requests",
              "escalated steps", "small-model steps", "thinking", "run"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row(run_id)) + " |" for run_id in run_ids]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Results table from run summaries.")
    parser.add_argument("--run", action="append", required=True, help="run id under runs/; repeatable")
    parser.add_argument("--output", type=Path, help="also write the table to this file")
    args = parser.parse_args()
    text = table(args.run)
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)


if __name__ == "__main__":
    main()
