"""The team without servers: the cards, the plan schema, dependency order, results passed between steps, the
6-step cap, uncovered parts, each specialist's tools, stopping at a failed step and the failure labels. Scripted
fake models play against a real Env on the multi-API practice cases."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agents.orchestrator import (
    MAX_PLAN_STEPS,
    Member,
    TeamResult,
    failure_label,
    plan_schema,
    run_order,
    run_team,
    setup_models,
    step_instruction,
)
from agents.specialist import active_names, inactive_names, load_specialists
from common.eval_config import SeatbeltConfig
from common.schemas import Call, Plan, PlanStep, Step
from common.tool_loop import ModelReply, StepResult, ToolCall, chat_tools, env_tools
from gym_env.env import Env, tool_schemas, tools_for
from gym_env.tasks import read_cases, system_prompt
from gym_env.verify import check

SEATBELT = SeatbeltConfig(max_retries_per_step=2, transient_retries=3, transient_backoff_s=0)
CASES = {case.id: case for case in read_cases(Path("data/practice/multi.jsonl"))}
SPECIALISTS = load_specialists()
TICKET = {"employee_id": "E1012", "category": "incident", "summary": "Payments crashing in prod after the deploy",
          "priority": "P1"}


def turn(*calls: tuple[str, Any], text: str = "") -> ModelReply:
    """A model reply: tool calls, or a final answer."""
    tool_calls = [ToolCall(f"call-{i}", name, json.dumps(args)) for i, (name, args) in enumerate(calls)]
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in tool_calls
        ]
    return ModelReply(message, text, tool_calls, prompt_tokens=100, completion_tokens=10, latency_s=0.0)


def done(text: str) -> list[ModelReply]:
    """A final answer and a self-check that finds no tool call still needed."""
    return [turn(text=text), turn(text=json.dumps({"reason": "done", "tool_call_needed": False}))]


def plan(*steps: tuple[str, str, list[int]], uncovered: tuple[str, ...] = (), reply: str = "") -> ModelReply:
    return turn(text=json.dumps({
        "analysis": "test plan",
        "steps": [{"agent": agent, "instruction": text, "depends_on": after} for agent, text, after in steps],
        "uncovered": list(uncovered),
        "reply": reply,
    }))


class ScriptedModel:
    """Replies in order; records every request."""

    def __init__(self, *replies: ModelReply) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, messages: list[dict[str, Any]], tools: Any, json_schema: Any) -> ModelReply:
        self.requests.append({"messages": list(messages), "tools": tools, "json_schema": json_schema})
        return self.replies.pop(0)

    def tool_names(self, request: int = 0) -> set[str]:
        return {tool["function"]["name"] for tool in self.requests[request]["tools"]}


def run(case_id: str, planner: ScriptedModel, seatbelt: SeatbeltConfig | None = SEATBELT,
        **models: ScriptedModel) -> tuple[TeamResult, Env]:
    case = CASES[case_id]
    env = Env(case.fixture, apis=case.apis)
    members = [Member(SPECIALISTS[name], model, f"{name}_r1") for name, model in models.items()]
    result = asyncio.run(run_team(case.request, system_prompt(case), chat_tools(tool_schemas(case.apis)), planner,
                                  "lightning", members, env_tools(env), max_steps=10, seatbelt=seatbelt))
    return result, env


def test_the_registry_lists_a_card_per_specialist_and_all_are_active() -> None:
    assert set(SPECIALISTS) == {"helpdesk", "ops"}
    assert active_names() == ["helpdesk", "ops"] and inactive_names() == []  # evals need every specialist
    for specialist in SPECIALISTS.values():
        assert tools_for([specialist.api])
        assert {"name", "description", "skills"} <= specialist.card.keys()  # A2A Agent Card fields
        assert all({"id", "name", "description"} <= skill.keys() for skill in specialist.card["skills"])


def test_the_plan_schema_allows_only_active_specialists_and_six_steps() -> None:
    schema = plan_schema(["helpdesk"])
    steps = schema["properties"]["steps"]
    assert steps["maxItems"] == MAX_PLAN_STEPS == 6
    assert steps["items"]["properties"]["agent"]["enum"] == ["helpdesk"]
    assert list(schema["properties"])[0] == "analysis"  # worked out before the steps
    assert plan_schema([])["properties"]["steps"]["maxItems"] == 0


def test_the_demo_request_opens_the_ticket_first_and_passes_its_id_to_ops() -> None:
    planner = ScriptedModel(plan(
        ("ops", "Roll payments in prod back to its last healthy revision, with step 2's ticket ID as change_ticket.", [2]),
        ("helpdesk", "Open a P1 incident ticket for the current user: payments in prod crashes after this morning's deploy.", []),
    ))
    helpdesk = ScriptedModel(turn(("create_ticket", TICKET)), *done("Opened HD-1042."))
    ops = ScriptedModel(
        turn(("rollout_history", {"namespace": "prod", "deployment": "payments"})),
        turn(("rollback", {"namespace": "prod", "deployment": "payments", "to_revision": 8, "change_ticket": "HD-1042"})),
        *done("Rolled payments back to revision 8."),
    )
    result, env = run("pr-multi-01", planner, helpdesk=helpdesk, ops=ops)

    assert result.order == [2, 1]  # the ticket step first, though the plan lists it second
    instruction = ops.requests[0]["messages"][1]["content"]
    assert "Results of the earlier steps" in instruction and '"id":"HD-1042"' in instruction
    assert helpdesk.tool_names() == {tool.name for tool in tools_for(["helpdesk"])}
    assert ops.tool_names() == {tool.name for tool in tools_for(["ops"])}
    assert helpdesk.requests[0]["messages"][0]["content"] == system_prompt(CASES["pr-multi-01"])  # the single setups'
    assert check(CASES["pr-multi-01"], env.state, result.answer).passed and not result.failed
    pods = env.call("get_pods", {"namespace": "prod", "deployment": "payments"}).output["pods"]
    assert {pod["status"] for pod in pods} == {"Running"}
    assert result.answer == "Opened HD-1042.\n\nRolled payments back to revision 8."
    assert planner.requests[0]["json_schema"] == plan_schema(["helpdesk", "ops"])
    assert result.prompt_tokens == 800 and [step.step.prompt_tokens for step in result.steps] == [300, 400]


def test_a_step_gets_only_the_results_of_the_steps_it_depends_on() -> None:
    the_plan = Plan(model="lightning", analysis="", uncovered=[], reply="", prompt_tokens=0, completion_tokens=0,
                    steps=[PlanStep(agent="helpdesk", instruction=f"step {n}", depends_on=after)
                           for n, after in ((1, []), (2, []), (3, [1, 1]))])
    calls = [Call(turn=1, tool="list_tickets", args={}, status=200, output={"tickets": ["HD-1046"]}),
             Call(turn=2, tool="get_ticket", args={}, status=404, output={"error": {"code": "not_found"}})]

    def result(n: int) -> StepResult:
        step = Step(agent="helpdesk", model="student", calls=calls if n == 1 else [], seatbelt_events=[])
        return StepResult(step, [], f"answer {n}", 0, 0, 0.0)

    text = step_instruction(3, the_plan, {1: result(1), 2: result(2)})
    assert text.startswith("step 3\n") and text.count("answer 1") == 1 and "answer 2" not in text
    assert '{"tool":"list_tickets","output":{"tickets":["HD-1046"]}}' in text and "not_found" not in text
    assert step_instruction(2, the_plan, {1: result(1)}) == "step 2"


def test_run_order_follows_dependencies_and_falls_back_to_plan_order() -> None:
    def steps(*after: list[int]) -> list[PlanStep]:
        return [PlanStep(agent="ops", instruction="", depends_on=a) for a in after]

    assert run_order(steps([], [1], [2])) == [0, 1, 2]
    assert run_order(steps([3], [], [2])) == [1, 2, 0]
    assert run_order(steps([2], [1], [])) == [2, 0, 1]  # a cycle runs in plan order after the free step
    assert run_order(steps([9, 1], [0])) == [0, 1]  # missing steps and self-dependencies are ignored


def test_only_active_specialists_are_planned_and_uncovered_parts_are_named() -> None:
    planner = ScriptedModel(plan(("helpdesk", "Open a P1 incident ticket for payments crashing in prod.", []),
                                 uncovered=("Roll back the payments deployment in prod.",)))
    helpdesk = ScriptedModel(turn(("create_ticket", TICKET)), *done("Opened HD-1042."))
    result, env = run("pr-multi-01", planner, helpdesk=helpdesk)

    request = planner.requests[0]
    assert request["json_schema"]["properties"]["steps"]["items"]["properties"]["agent"]["enum"] == ["helpdesk"]
    assert SPECIALISTS["helpdesk"].card["description"] in request["messages"][0]["content"]
    assert SPECIALISTS["ops"].card["description"] not in request["messages"][0]["content"]
    assert result.answer.endswith("Not done, because no specialist on the team covers it: "
                                  "Roll back the payments deployment in prod.")
    assert not check(CASES["pr-multi-01"], env.state, result.answer).passed  # the rollback didn't happen


def test_the_plan_runs_at_most_six_steps() -> None:
    planner = ScriptedModel(plan(*[("helpdesk", f"Look up the current user's devices ({n}).", []) for n in range(8)]))
    helpdesk = ScriptedModel(*[reply for _ in range(MAX_PLAN_STEPS) for reply in done("No changes needed.")])
    result, _ = run("pr-multi-08", planner, helpdesk=helpdesk)
    assert result.order == [1, 2, 3, 4, 5, 6] and not helpdesk.replies
    assert "only the first 6 ran" in result.answer


def test_the_team_stops_at_the_first_failed_step() -> None:
    planner = ScriptedModel(plan(("helpdesk", "Open a P1 incident ticket.", []),
                                 ("ops", "Roll back payments in prod with step 1's ticket.", [1])))
    helpdesk = ScriptedModel(turn(*[("drop_tickets", {})] * 3))  # three invalid calls: no retries left
    ops = ScriptedModel()
    result, env = run("pr-multi-01", planner, helpdesk=helpdesk, ops=ops)
    assert result.failed and result.order == [1] and not ops.requests
    assert "Step 1 (helpdesk) failed, so the team stopped there." in result.answer
    assert env.state.tickets == Env("incident_prod_payments").state.tickets


@pytest.mark.parametrize("seatbelt", [SEATBELT, None], ids=["seatbelt", "no seatbelt"])
def test_a_specialist_cannot_call_another_api_s_tools(seatbelt: SeatbeltConfig | None) -> None:
    planner = ScriptedModel(plan(("helpdesk", "Roll back payments in prod.", [])))
    rollback = {"namespace": "prod", "deployment": "payments", "to_revision": 8, "change_ticket": "HD-1041"}
    helpdesk = ScriptedModel(turn(("rollback", rollback)), *done("I can't roll back deployments."))
    result, env = run("pr-multi-01", planner, seatbelt=seatbelt, helpdesk=helpdesk)
    (call,) = result.steps[0].step.calls
    assert call.status in (0, 404) and call.output["error"]["code"] == "unknown_tool"
    assert "list_devices" in call.output["error"]["message"] and "get_pods" not in call.output["error"]["message"]
    assert env.calls == [] and not env.state.changes


def test_a_reply_without_steps_is_the_answer() -> None:
    planner = ScriptedModel(plan(reply="You have no tickets. Which deployment should I roll back, and in which namespace?"))
    result, env = run("pr-multi-07", planner, helpdesk=ScriptedModel(), ops=ScriptedModel())
    assert result.steps == [] and not result.failed
    assert result.answer.startswith("You have no tickets.")
    assert check(CASES["pr-multi-07"], env.state, result.answer).passed


@pytest.mark.parametrize("text, error", [
    ("not a plan", "isn't a valid plan"),
    (json.dumps({"analysis": "", "steps": [{"agent": "calendar", "instruction": "Book a room.", "depends_on": []}],
                 "uncovered": [], "reply": ""}), "aren't active"),
])
def test_without_a_valid_plan_nothing_runs_and_the_attempt_fails(text: str, error: str) -> None:
    helpdesk = ScriptedModel()
    result, env = run("pr-multi-08", ScriptedModel(turn(text=text)), helpdesk=helpdesk)
    assert error in result.plan.error and result.failed and not helpdesk.requests and env.calls == []
    assert "couldn't be planned" in result.answer


def test_failure_labels_compare_the_plan_s_agents_with_the_gold_apis() -> None:
    def planned(*agents: str) -> Plan:
        return Plan(model="lightning", analysis="", uncovered=[], reply="", prompt_tokens=0, completion_tokens=0,
                    steps=[PlanStep(agent=agent, instruction="", depends_on=[]) for agent in agents])

    assert failure_label(planned("helpdesk", "ops", "ops"), {"helpdesk", "ops"}, SPECIALISTS) == "specialist"
    assert failure_label(planned("helpdesk"), {"helpdesk", "ops"}, SPECIALISTS) == "routing"
    assert failure_label(planned("ops"), set(), SPECIALISTS) == "routing"  # acted on a case that changes nothing
    assert failure_label(planned(), set(), SPECIALISTS) == "specialist"
    assert failure_label(None, {"ops"}, SPECIALISTS) == "routing"
    assert failure_label(planned("ops").model_copy(update={"error": "invalid"}), {"ops"}, SPECIALISTS) == "routing"


def test_team_setups_differ_only_in_the_specialists_models() -> None:
    assert setup_models("team_base") == ("lightning", {"helpdesk": "student", "ops": "student"})
    assert setup_models("team_trained") == ("lightning", {"helpdesk": "helpdesk_r1", "ops": "ops_r1"})
