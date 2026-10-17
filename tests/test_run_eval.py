"""eval/run_eval.py without servers: the refusals, and turning rollouts into traces and a summary.
Nothing here reads data/test/ (hard rule 4)."""

import argparse
import sys
from typing import Any

import pytest
import yaml

from common.eval_config import load_eval_config
from common.schemas import Case
from eval import run_eval

CONFIG = load_eval_config()


def args(**overrides: Any) -> argparse.Namespace:
    return argparse.Namespace(**({"split": "practice", "final": False, "chaos": False} | overrides))


def test_refuses_replay_mode() -> None:
    assert "replay" in run_eval.refusal(args(), {"LLM_MODE": "replay"}, CONFIG, server_chaos=False)
    assert run_eval.refusal(args(), {"LLM_MODE": "record"}, CONFIG, server_chaos=False) is None


def test_refuses_the_test_split_without_final() -> None:
    assert "locked" in run_eval.refusal(args(split="test"), {}, CONFIG, server_chaos=False)
    assert run_eval.refusal(args(split="test", final=True), {}, CONFIG, server_chaos=False) is None


def test_refuses_chaos_unless_flagged() -> None:
    chaotic = CONFIG.model_copy(update={"chaos": True})
    assert "--chaos" in run_eval.refusal(args(), {}, chaotic, server_chaos=False)
    assert "--chaos" in run_eval.refusal(args(), {}, CONFIG, server_chaos=True)
    assert run_eval.refusal(args(chaos=True), {}, chaotic, server_chaos=True) is None


def test_shipped_configs_have_chaos_off() -> None:
    assert not CONFIG.chaos and not run_eval.server_chaos_enabled()


def test_main_refuses_the_test_split_before_reading_any_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_reads(*_: Any) -> None:
        raise AssertionError("cases were read")

    monkeypatch.setattr(run_eval, "load_cases", no_reads)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--setup", "base_seatbelt", "--api", "ops", "--split", "test"])
    with pytest.raises(SystemExit, match="locked"):
        run_eval.main()


def test_setups_are_the_gym_configs() -> None:
    assert run_eval.SETUPS == ["base_raw", "base_seatbelt", "lightning", "r1"]


def test_r1_is_base_seatbelt_with_the_api_s_own_adapter() -> None:
    r1, base = (yaml.safe_load((run_eval.GYM_CONFIGS / f"{setup}.yaml").read_text()) for setup in ("r1", "base_seatbelt"))
    assert r1 | {"policy_model_name": "x"} == base | {"policy_model_name": "x"}  # hard rule 7: only the model differs
    assert run_eval.adapter_model("r1", "helpdesk") == "helpdesk_r1"
    assert run_eval.adapter_model("r1", "ops") == "ops_r1"
    assert run_eval.adapter_model("base_seatbelt", "ops") is None


def test_the_adapter_name_reaches_the_agent_and_the_model_server() -> None:
    omegaconf = pytest.importorskip("omegaconf")
    files = [omegaconf.OmegaConf.load(path) for path in run_eval.gym_configs("r1")]
    assert omegaconf.OmegaConf.is_missing(omegaconf.OmegaConf.merge(*files), "policy_model_name")  # Gym stops without it
    config = omegaconf.OmegaConf.merge(*files, omegaconf.OmegaConf.from_dotlist(["policy_model_name=ops_r1"]))
    assert config.itops_agent.responses_api_agents.itops_agent.model_name == "ops_r1"
    assert config.policy_model.responses_api_models.vllm_model.model == "ops_r1"


def case(case_id: str) -> Case:
    return Case(id=case_id, request="Scale checkout in staging to 3.", user_id="E1011", fixture="default",
                apis=["ops"], gold_calls=[{"api": "ops", "tool": "scale",
                                           "args": {"namespace": "staging", "deployment": "checkout", "replicas": 3}}],
                kind="single")


def rollout(case_id: str, attempt: int, reward: float, model: str = "student") -> dict[str, Any]:
    step = {"agent": "single", "model": model, "calls": [], "seatbelt_events": []}
    return {"task_id": case_id, "_ng_rollout_index": attempt - 1, "reward": reward,
            "trace": {"steps": [step], "prompt_tokens": 1_000_000, "completion_tokens": 100_000, "latency_s": 2.0}}


def test_a_case_passes_on_two_of_three_attempts_and_missing_attempts_fail() -> None:
    rows = [rollout("a", 1, 1.0), rollout("a", 2, 0.0), rollout("a", 3, 1.0),  # 2/3: pass
            rollout("b", 1, 1.0), rollout("b", 2, 0.0), rollout("b", 3, 0.0),  # 1/3: fail
            rollout("c", 1, 1.0)]                                              # 2 attempts missing: fail
    traces = [run_eval.to_trace(row, "run", "base_seatbelt", CONFIG) for row in rows]
    summary, per_case = run_eval.summarize(traces, [case("a"), case("b"), case("c")], "base_seatbelt", CONFIG)

    assert summary.root["base_seatbelt"].passes == "1/3"
    assert per_case["a"] == {"kind": "single", "passed_attempts": "2/3", "collected_attempts": 3, "passed": True}
    assert per_case["c"]["passed_attempts"] == "1/3" and per_case["c"]["collected_attempts"] == 1
    assert traces[2].attempt == 3 and traces[2].passed and not traces[1].passed


def test_cost_uses_the_model_s_price() -> None:
    student, lightning = (run_eval.to_trace(rollout("a", 1, 1.0, model), "run", "lightning", CONFIG)
                          for model in ("student", "lightning"))
    prices = CONFIG.prices_usd_per_1m_tokens
    assert student.cost_usd == pytest.approx(prices["student"].prompt + 0.1 * prices["student"].completion)
    assert lightning.cost_usd == pytest.approx(prices["lightning"].prompt + 0.1 * prices["lightning"].completion)
    adapter = run_eval.to_trace(rollout("a", 1, 1.0, "ops_r1"), "run", "r1", CONFIG)
    assert adapter.cost_usd == student.cost_usd  # adapters are priced as the student
