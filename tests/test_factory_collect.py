"""factory/collect.py without servers: the agent config, the refusals, and the run folder it writes."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from common.eval_config import EVAL_CONFIG
from common.schemas import Case
from factory import collect
from factory.sampler import load_factory_config

FACTORY = load_factory_config()


def test_agent_config_is_the_eval_config_with_only_the_collect_fields_changed() -> None:
    settings = collect.agent_config(FACTORY)
    base = yaml.safe_load(EVAL_CONFIG.read_text())
    changed = {key for key in base if settings[key] != base[key]}
    assert changed <= set(FACTORY["collect"]) and settings.keys() == base.keys()
    assert settings["thinking"] is True and settings["attempts_per_case"] == 1
    assert settings["seatbelt"] == base["seatbelt"] and settings["max_steps"] == base["max_steps"]


def test_refuses_test_data_and_chaos() -> None:
    settings = collect.agent_config(FACTORY)
    assert "locked" in collect.refusal(collect.REPO / "data" / "test" / "x.jsonl", settings, server_chaos=False)
    assert "chaos" in collect.refusal(Path("t.jsonl"), settings | {"chaos": True}, server_chaos=False)
    assert "chaos" in collect.refusal(Path("t.jsonl"), settings, server_chaos=True)
    assert collect.refusal(Path("t.jsonl"), settings, server_chaos=False) is None


def test_main_writes_the_run_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    case = Case(id="ops-t-0001", request="Scale checkout in staging to 3 replicas.", user_id="E1011", fixture="default",
                apis=["ops"], kind="single",
                gold_calls=[{"api": "ops", "tool": "scale", "args": {"namespace": "staging", "deployment": "checkout", "replicas": 3}}])
    tasks = tmp_path / "tasks_t.jsonl"
    tasks.write_text(case.model_dump_json() + "\n")
    started: dict[str, Any] = {}

    def start(setup: str, config_path: Path, chaos: bool, log: Path) -> str:
        started.update(setup=setup, config=yaml.safe_load(config_path.read_text()), chaos=chaos)
        return "servers"

    def run(rows: Path, rollouts: Path, attempts: int, log: Path) -> int:
        rollouts.write_text(json.dumps({"task_id": "ops-t-0001", "reward": 1.0}) + "\n")
        return 0

    monkeypatch.setattr(collect, "REPO", tmp_path)
    monkeypatch.setattr(collect, "server_chaos_enabled", lambda: False)
    monkeypatch.setattr(collect, "start_servers", start)
    monkeypatch.setattr(collect, "collect", run)
    monkeypatch.setattr(collect, "stop_servers", lambda servers: started.update(stopped=servers))
    monkeypatch.setattr(sys, "argv", ["collect", "--tasks", str(tasks), "--run-id", "c1"])
    collect.main()

    run_dir = tmp_path / "runs" / "c1"
    config = json.loads((run_dir / "config.json").read_text())
    assert config["tasks"] == "tasks_t.jsonl" and config["setup"] == "lightning" and config["task_count"] == 1
    assert started == {"setup": "lightning", "config": collect.agent_config(FACTORY), "chaos": False, "stopped": "servers"}
    rows = [json.loads(line) for line in (run_dir / "tasks.jsonl").read_text().splitlines()]
    assert rows[0]["task_id"] == "ops-t-0001" and rows[0]["case"]["request"] == case.request
