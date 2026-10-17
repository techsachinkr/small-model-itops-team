"""factory/sampler.py: every sampled task is usable, sampling is reproducible, and the phrasing checks
catch what they should. No model involved."""

import random
import re

import pytest

from common.schemas import Case
from factory import sampler
from factory.sampler import GROUPS, Mention, SampledTask, load_factory_config, sample, task_problems
from gym_env.verify import check

WEIGHTS = load_factory_config()["tasks"]["sampled_kinds"]
LOOKUPS = {"list_devices", "list_tickets", "rollout_history", "list_deployments"}


@pytest.fixture(scope="module", params=["helpdesk", "ops"])
def tasks(request: pytest.FixtureRequest) -> list[SampledTask]:
    return sample(request.param, 300, random.Random(0), WEIGHTS)


def test_every_template_group_exists_for_both_apis() -> None:
    for api in ("helpdesk", "ops"):
        for group in GROUPS:
            assert sampler.templates_for(api, group), (api, group)


def test_sampled_tasks_are_valid_cases(tasks: list[SampledTask]) -> None:
    for i, drawn in enumerate(tasks):
        case = drawn.case(f"t-{i}", "request")
        assert task_problems(case) == [], (drawn.template, drawn.brief)
        assert case.kind == drawn.kind


def test_gold_runs_pass_the_verifier_and_doing_nothing_passes_only_without_changes(tasks: list[SampledTask]) -> None:
    from gym_env.env import Env
    from gym_env.verify import expected_state

    for i, drawn in enumerate(tasks[:80]):
        case = drawn.case(f"t-{i}", "request")
        assert check(case, expected_state(case), "").passed
        idle = check(case, Env(case.fixture, apis=case.apis).state, "")
        assert idle.passed == (case.kind != "single"), drawn.template


def test_steps_match_the_group(tasks: list[SampledTask]) -> None:
    for drawn in tasks:
        tools = [call.tool for call in drawn.gold_calls]
        if drawn.group == "single_1_step":
            assert len(tools) == 1 and tools[0] not in LOOKUPS, drawn.template
        elif drawn.group == "single_2_steps":
            assert len(tools) == 2 and tools[0] in LOOKUPS and tools[1] not in LOOKUPS, drawn.template
        else:
            assert tools == [], drawn.template


def test_every_group_is_sampled_and_tasks_are_distinct(tasks: list[SampledTask]) -> None:
    assert {drawn.group for drawn in tasks} == set(GROUPS)
    keys = [drawn.model_dump_json(include={"template", "user_id", "fixture", "brief"}) for drawn in tasks]
    assert len(keys) == len(set(keys))


def test_patterns_compile(tasks: list[SampledTask]) -> None:
    for drawn in tasks:
        for mention in drawn.include + drawn.leave_out:
            re.compile(mention.pattern)


def test_sampling_is_reproducible() -> None:
    first = sample("ops", 50, random.Random(7), WEIGHTS)
    again = sample("ops", 50, random.Random(7), WEIGHTS)
    other = sample("ops", 50, random.Random(8), WEIGHTS)
    assert first == again and first != other


def test_two_step_tasks_hide_the_id_the_lookup_finds() -> None:
    drawn = next(t for t in sample("helpdesk", 200, random.Random(0), WEIGHTS) if t.template == "hardware_ticket_lookup_device")
    device_id = drawn.gold_calls[1].args["device_id"]
    assert drawn.phrasing_problems(f"My laptop {device_id} broke") != []
    priority = drawn.gold_calls[1].args["priority"]
    assert drawn.phrasing_problems(f"My laptop is broken, please open a {priority} ticket.") == []


def test_phrasing_checks() -> None:
    drawn = SampledTask(api="ops", group="single_1_step", template="scale_to", user_id="E1011", fixture="default",
                        gold_calls=[], brief="",
                        include=[Mention(text="4", pattern=sampler.count(4)), Mention(text="prod", pattern=sampler.NAMESPACE["prod"])],
                        leave_out=[sampler.ANY_TICKET_ID])
    assert drawn.phrasing_problems("Scale checkout in production to four replicas.") == []
    assert drawn.phrasing_problems("Scale checkout in prod to 4, ticket HD-1041.") == ["it must leave out any ticket ID"]
    assert drawn.phrasing_problems("Scale checkout in staging to 14.") == ["it must include 4", "it must include prod"]
    assert re.search(sampler.version("2.3.1"), "back to v2.3.1.", re.I)
    assert not re.search(sampler.version("2.3.1"), "back to 2.3.10", re.I)
    assert not re.search(sampler.count(8), "HD-1048 and 2.4.8-rc", re.I)


def test_task_problems_catches_bad_tasks() -> None:
    def case(**fields: object) -> Case:
        return Case.model_validate({"id": "x", "request": "r", "user_id": "E1011", "fixture": "default",
                                    "apis": ["helpdesk"], "kind": "single", **fields})

    failing = case(gold_calls=[{"api": "helpdesk", "tool": "get_ticket", "args": {"ticket_id": "HD-9999"}}])
    assert "failed" in task_problems(failing)[0]
    read_only = [{"api": "helpdesk", "tool": "list_devices", "args": {"employee_id": "E1001"}}]
    assert task_problems(case(gold_calls=read_only)) == ["the gold calls change nothing"]
    assert task_problems(case(kind="read_only", gold_calls=read_only, answer_must_contain=["X1 Carbon"])) == []
    assert "no gold call returns" in task_problems(case(kind="read_only", gold_calls=read_only, answer_must_contain=["iPad"]))[0]
    leaked = case(kind="read_only", request="Is my X1 Carbon listed?", gold_calls=read_only, answer_must_contain=["X1 Carbon"])
    assert "already in the request" in task_problems(leaked)[0]
