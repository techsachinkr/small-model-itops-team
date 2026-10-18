"""eval/run_eval.py without servers: the refusals, and turning rollouts into traces and a summary.
Nothing here reads data/test/ (hard rule 4)."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from common.eval_config import load_eval_config
from common.schemas import Case
from eval import run_eval
from gym_env.env import tool_schemas
from gym_env.tasks import case_to_row, read_cases

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


def test_refuses_team_runs_with_a_specialist_switched_off() -> None:
    assert "switched off (ops)" in run_eval.refusal(args(), {}, CONFIG, server_chaos=False, inactive_specialists=["ops"])
    assert run_eval.refusal(args(), {}, CONFIG, server_chaos=False, inactive_specialists=[]) is None


@pytest.mark.parametrize("argv, message", [
    (["--setup", "team_base", "--api", "ops", "--split", "practice"], "drop --api"),
    (["--setup", "single_base", "--api", "ops", "--split", "practice"], "drop --api"),
    (["--setup", "r1", "--split", "practice"], "needs --api"),
    (["--setup", "team_base", "--split", "test"], "locked"),  # no --api needed; refused before reading cases
])
def test_system_setups_take_no_api_and_ladder_setups_need_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str], message: str
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_eval", *argv])
    with pytest.raises(SystemExit) as exit_info:
        run_eval.main()
    assert message in str(exit_info.value) + capsys.readouterr().err


def test_setups_are_the_gym_configs() -> None:
    assert run_eval.SETUPS == ["base_raw", "base_seatbelt", "lightning", "r1", "single_base", "single_lightning",
                               "single_trained", "team_base", "team_trained"]


def test_system_setups_run_on_the_multi_api_cases_of_every_file() -> None:
    cases = run_eval.load_cases("practice", None)
    assert {case.id for case in cases} == {case.id for case in read_cases(Path("data/practice/multi.jsonl"))}
    assert all(len(case.apis) > 1 for case in cases)
    assert all(case.apis == ["ops"] for case in run_eval.load_cases("practice", "ops"))


def test_team_setups_run_the_team_agent_and_differ_only_in_the_specialists_models() -> None:
    assert run_eval.agent_name("team_base") == "itops_team" and run_eval.agent_name("r1") == "itops_agent"
    assert [path.name for path in run_eval.gym_configs("team_trained")] == ["itops.yaml", "eval_team.yaml", "team_trained.yaml"]
    base, trained = (yaml.safe_load((run_eval.GYM_CONFIGS / f"{setup}.yaml").read_text()) for setup in ("team_base", "team_trained"))
    assert base.keys() == trained.keys() == {"helpdesk_model_name", "ops_model_name"}  # hard rule 7


def test_the_specialists_models_reach_the_team_agent_and_its_model_servers() -> None:
    omegaconf = pytest.importorskip("omegaconf")
    files = [omegaconf.OmegaConf.load(path) for path in run_eval.gym_configs("team_trained")]
    config = omegaconf.OmegaConf.merge(*files)
    agent = config.itops_team.responses_api_agents.itops_team
    assert (agent.planner_model_name, agent.helpdesk_model_name, agent.ops_model_name) == ("lightning", "helpdesk_r1", "ops_r1")
    servers = {name: config[name].responses_api_models.vllm_model for name in ("planner_model", "helpdesk_model", "ops_model")}
    assert {name: (server.model, server.base_url) for name, server in servers.items()} == {
        "planner_model": ("lightning", "http://localhost:8000/v1"),
        "helpdesk_model": ("helpdesk_r1", "http://localhost:8001/v1"),
        "ops_model": ("ops_r1", "http://localhost:8001/v1"),
    }
    single = omegaconf.OmegaConf.merge(*[omegaconf.OmegaConf.load(path) for path in run_eval.gym_configs("lightning")])
    for key in ("return_token_id_information", "uses_reasoning_parser"):  # the model servers match the single setups'
        assert all(server[key] == single.policy_model.responses_api_models.vllm_model[key] for server in servers.values())


def test_r1_is_base_seatbelt_with_the_api_s_own_adapter() -> None:
    r1, base = (yaml.safe_load((run_eval.GYM_CONFIGS / f"{setup}.yaml").read_text()) for setup in ("r1", "base_seatbelt"))
    assert r1 | {"policy_model_name": "x"} == base | {"policy_model_name": "x"}  # hard rule 7: only the model differs
    assert run_eval.adapter_model("r1", "helpdesk") == "helpdesk_r1"
    assert run_eval.adapter_model("r1", "ops") == "ops_r1"
    assert run_eval.adapter_model("base_seatbelt", "ops") is None


def test_single_setups_are_the_ladder_s_models_with_every_tool() -> None:
    """One model with every tool and no orchestrator: base_seatbelt's and lightning's files, and single_base's with
    the all_r1 adapter (priced as the student), run by the single agent on the multi-API cases, whose rows carry every
    API's tools in a fixed order."""
    single_base, base, single_lightning, lightning, single_trained = (
        yaml.safe_load((run_eval.GYM_CONFIGS / f"{setup}.yaml").read_text())
        for setup in ("single_base", "base_seatbelt", "single_lightning", "lightning", "single_trained"))
    assert single_base == base and single_lightning == lightning  # hard rule 7: only the model differs
    assert single_trained == single_base | {"policy_model_name": "all_r1"}
    assert CONFIG.prices_for("all_r1") == CONFIG.prices_for("student")
    for setup in ("single_base", "single_lightning", "single_trained"):
        assert run_eval.is_system(setup) and run_eval.agent_name(setup) == "itops_agent"
        assert [path.name for path in run_eval.gym_configs(setup)] == ["itops.yaml", "eval.yaml", f"{setup}.yaml"]
    every_tool = tool_schemas(run_eval.APIS)
    assert len(every_tool) > len(tool_schemas(["helpdesk"])) > 0
    for case in run_eval.load_cases("practice", None):
        assert case_to_row(case)["responses_create_params"]["tools"] == every_tool


@pytest.fixture
def fake_gym(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stands in for `gym env start` and returns how it was called."""
    started: dict[str, Any] = {}

    class Gym:  # stands in for `gym env start`: records how it was called and reports its servers ready
        def __init__(self, command: list[str], env: dict[str, str], stdout: Any, **kwargs: Any) -> None:
            started.update(command=command, env=env)
            stdout.write("All 4 / 4 servers ready\n")
            stdout.flush()

    class FreePort:
        def __enter__(self) -> "FreePort":
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def connect_ex(self, address: tuple[str, int]) -> int:
            return 1  # nothing listening

    monkeypatch.setattr(run_eval.subprocess, "Popen", Gym)
    monkeypatch.setattr(run_eval.socket, "socket", FreePort)
    return started


def test_gym_starts_with_the_adapter_and_without_ray_s_uv_hook(fake_gym: dict[str, Any], tmp_path: Path) -> None:
    run_eval.start_servers("r1", run_eval.EVAL_CONFIG, False, tmp_path / "gym_env.log", "helpdesk_r1")
    assert "++policy_model_name=helpdesk_r1" in fake_gym["command"]
    assert "++itops_agent.responses_api_agents.itops_agent.eval_config=configs/eval.yaml" in fake_gym["command"]
    # Under `uv run`, Ray's uv hook stopped every Gym server at startup (task 2.2).
    assert fake_gym["env"]["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"


def test_team_runs_start_and_collect_with_the_team_agent(
    fake_gym: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_eval.start_servers("team_base", run_eval.EVAL_CONFIG, False, tmp_path / "gym_env.log")
    assert "gym_configs/eval_team.yaml" in fake_gym["command"] and "gym_configs/team_base.yaml" in fake_gym["command"]
    assert "++itops_team.responses_api_agents.itops_team.eval_config=configs/eval.yaml" in fake_gym["command"]
    collected: dict[str, Any] = {}
    monkeypatch.setattr(run_eval.subprocess, "run",
                        lambda command, **kwargs: collected.update(command=command) or SimpleNamespace(returncode=0))
    run_eval.collect(tmp_path / "tasks.jsonl", tmp_path / "rollouts.jsonl", 3, tmp_path / "gym_eval.log", "itops_team")
    assert collected["command"][collected["command"].index("--agent") + 1] == "itops_team"


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


def test_steps_that_carry_their_tokens_are_priced_the_same() -> None:
    old = run_eval.to_trace(rollout("a", 1, 1.0), "run", "base_seatbelt", CONFIG)
    row = rollout("a", 1, 1.0)
    row["trace"]["steps"][0] |= {"prompt_tokens": 1_000_000, "completion_tokens": 100_000}
    assert run_eval.to_trace(row, "run", "base_seatbelt", CONFIG).cost_usd == pytest.approx(old.cost_usd)


def team_rollout(reward: float, agents: list[str], gold_apis: list[str]) -> dict[str, Any]:
    """A team attempt: the planner on Lightning (1M prompt tokens), then one step per agent on its adapter."""
    plan = {"model": "lightning", "analysis": "", "uncovered": [], "reply": "", "prompt_tokens": 1_000_000,
            "completion_tokens": 0, "steps": [{"agent": agent, "instruction": "", "depends_on": []} for agent in agents]}
    steps = [{"agent": agent, "model": f"{agent}_r1", "calls": [], "seatbelt_events": [], "prompt_tokens": 1_000_000,
              "completion_tokens": 100_000} for agent in agents]
    return {"task_id": "pr-multi-01", "_ng_rollout_index": 0, "reward": reward,
            "case": {"gold_calls": [{"api": api, "tool": "any", "args": {}} for api in gold_apis]},
            "trace": {"plan": plan, "steps": steps, "prompt_tokens": 1_000_000 * (1 + len(agents)),
                      "completion_tokens": 100_000 * len(agents), "latency_s": 3.0}}


def test_team_attempts_price_each_model_s_tokens_and_label_failures() -> None:
    prices = CONFIG.prices_usd_per_1m_tokens
    student = prices["student"].prompt + 0.1 * prices["student"].completion
    passed = run_eval.to_trace(team_rollout(1.0, ["helpdesk", "ops"], ["helpdesk", "ops"]), "run", "team_trained", CONFIG)
    assert passed.cost_usd == pytest.approx(prices["lightning"].prompt + 2 * student)
    assert passed.plan is not None and passed.plan.agents == {"helpdesk", "ops"} and passed.failure_label is None
    routing = run_eval.to_trace(team_rollout(0.0, ["helpdesk"], ["helpdesk", "ops"]), "run", "team_trained", CONFIG)
    specialist = run_eval.to_trace(team_rollout(0.0, ["ops", "helpdesk"], ["helpdesk", "ops"]), "run", "team_base", CONFIG)
    assert (routing.failure_label, specialist.failure_label) == ("routing", "specialist")
    assert run_eval.to_trace(rollout("a", 1, 0.0), "run", "base_seatbelt", CONFIG).failure_label is None


@pytest.mark.parametrize("setup", run_eval.SETUPS)
def test_gym_resolves_every_setup_s_config(setup: str) -> None:
    """The merged config resolves as `gym env start` resolves it, offline (no servers, ports or network). A team
    setup once failed here: itops.yaml's smoke agent pointed at a model server only eval.yaml defines."""
    pytest.importorskip("nemo_gym")
    from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
    from omegaconf import DictConfig

    paths = [str(path.relative_to(run_eval.REPO)) for path in run_eval.gym_configs(setup)]
    overrides = {"policy_model_name": "helpdesk_r1"} if setup in run_eval.ADAPTER_SETUPS else {}
    config = GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(
        initial_global_config_dict=DictConfig({"config_paths": paths, **overrides}),
        skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True,
    ))
    assert run_eval.agent_name(setup) in config
