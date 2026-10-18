"""eval/report.py on made-up run folders: one row per run, partial runs marked, and team_escalate's escalation
columns."""

import json
from pathlib import Path

import pytest

from eval import report


def write_run(runs: Path, run_id: str, setup: str, attempts: int, small_model_steps: str | None = None) -> None:
    run_dir = runs / run_id
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "args": {"api": None, "split": "practice"}, "cases": 8,
        "eval_config": {"attempts_per_case": 3, "thinking": False},
    }))
    summary = {"passes": "6/8", "mean_latency_s": 16.66, "cost_per_1k_usd": 0.587, "small_model_steps": small_model_steps}
    (run_dir / "summary.json").write_text(json.dumps({setup: summary}))
    (run_dir / "traces.jsonl").write_text("{}\n" * attempts)


def test_rows_show_escalations_for_team_escalate_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report, "RUNS", tmp_path)
    write_run(tmp_path, "trained", "team_trained", 24)
    write_run(tmp_path, "escalate", "team_escalate", 23, small_model_steps="54/56")
    header, _, trained, escalate = report.table(["trained", "escalate"]).splitlines()
    columns = [cell.strip() for cell in header.strip("|").split("|")]
    assert columns[7:9] == ["escalated steps", "small-model steps"]
    assert trained == "| team_trained | all | practice | 6/8 | 24/24 | 16.7 | 0.59 | - | - | off | trained |"
    assert escalate == "| team_escalate | all | practice | 6/8 (partial) | 23/24 | 16.7 | 0.59 | 2 | 54/56 | off | escalate |"
