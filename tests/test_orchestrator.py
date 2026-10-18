"""The team without servers: the cards, the plan schema, dependency order, results passed between steps, the
6-step cap, uncovered parts, each specialist's tools, stopping at a failed step, escalation (team_escalate) and the
failure labels. Scripted fake models play against a real Env on the multi-API practice cases."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agents.orchestrator import (
    MAX_PLAN_STEPS,
    Escalation,
    Member,
    TeamResult,
    escalation_model,
    failure_label,
    plan_schema,
    run_order,
    run_team,
    setup_models,
    step_instruction,
    team_setups,
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
        escalation: ScriptedModel | None = None, max_steps: int = 10, **models: ScriptedModel) -> tuple[TeamResult, Env]:
    """`escalation`: the model that finishes steps the seatbelt gave up on (team_escalate); None, as in every other
    setup, leaves a given-up step failed."""
    case = CASES[case_id]
    env = Env(case.fixture, apis=case.apis)
    members = [Member(SPECIALISTS[name], model, f"{name}_r1") for name, model in models.items()]
    lightning = Escalation(escalation, "lightning") if escalation is not None else None
    result = asyncio.run(run_team(case.request, system_prompt(case), chat_tools(tool_schemas(case.apis)), planner,
                                  "lightning", members, env_tools(env), max_steps=max_steps, seatbelt=seatbelt,
                                  escalation=lightning))
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
    # Without escalation (every setup but team_escalate, hard rule 7) nobody takes the step over.
    step = result.steps[0].step
    assert step.gave_up and not step.escalated and step.escalation is None


DEMO_PLAN = (("helpdesk", "Open a P1 incident ticket for the current user: payments in prod crashes after the deploy.", []),
             ("ops", "Roll payments in prod back to its last healthy revision, with step 1's ticket ID as change_ticket.",
              [1]))
ROLLBACK = {"namespace": "prod", "deployment": "payments", "to_revision": 8, "change_ticket": "HD-1042"}


def ops_rolls_back() -> ScriptedModel:
    return ScriptedModel(turn(("rollout_history", {"namespace": "prod", "deployment": "payments"})),
                         turn(("rollback", ROLLBACK)), *done("Rolled payments back to revision 8."))


def opens_the_ticket_then_gives_up(cause: str) -> tuple[ScriptedModel, int]:
    """A helpdesk specialist that opens HD-1042 and then makes the seatbelt give up on its step, and the max steps
    that takes."""
    opened = turn(("create_ticket", TICKET))
    if cause == "retry limit":
        # The third invalid call uses up the last retry, so the valid call after it is never executed.
        note = {"ticket_id": "HD-1042", "note": "Rolling payments back."}
        return ScriptedModel(opened, turn(*[("drop_tickets", {})] * 3, ("update_ticket", note))), 10
    if cause == "self-check":
        needed = turn(text=json.dumps({"reason": "the rollback isn't done yet", "tool_call_needed": True}))
        return ScriptedModel(opened, *[reply for _ in range(3) for reply in (turn(text="Opened HD-1042."), needed)]), 10
    # max turns: still looking things up when the step's turns run out
    return ScriptedModel(opened, turn(("get_ticket", {"ticket_id": "HD-1042"})),
                         turn(("list_tickets", {"employee_id": "E1012"}))), 3


@pytest.mark.parametrize("cause", ["retry limit", "self-check", "max turns"])
def test_an_escalated_step_never_repeats_an_executed_call(cause: str) -> None:
    """CLAUDE.md "Escalation": Lightning gets the step's conversation so far, with the calls already executed and
    their results, and finishes the step with the same tools and seatbelt. It continues the step and never restarts
    it, so every call runs in the environment once, in the order it was made, and a call that was never executed
    stays that way."""
    helpdesk, max_steps = opens_the_ticket_then_gives_up(cause)
    # Lightning has its own retries: one invalid call, then a look-up, then its answer.
    lightning = ScriptedModel(turn(("drop_tickets", {})), turn(("get_ticket", {"ticket_id": "HD-1042"})),
                              *done("HD-1042 is open for the payments incident."))
    ops = ops_rolls_back()
    result, env = run("pr-multi-01", ScriptedModel(plan(*DEMO_PLAN)), escalation=lightning, max_steps=max_steps,
                      helpdesk=helpdesk, ops=ops)

    step = result.steps[0].step
    assert step.seatbelt_events[-1].kind == "gave_up" and step.escalated and step.escalation is not None
    assert (step.agent, step.model, step.escalation.agent, step.escalation.model) == (
        "helpdesk", "helpdesk_r1", "helpdesk", "lightning")
    assert not step.gave_up and not result.failed and result.order == [1, 2]  # Lightning finished it; ops ran

    # Lightning continues the step's conversation, with the specialist's tools and a fresh seatbelt.
    sent = lightning.requests[0]
    last_turn = [request for request in helpdesk.requests if request["json_schema"] is None][-1]["messages"]
    assert sent["messages"][:len(last_turn)] == last_turn
    assert sent["messages"][-1]["role"] == "user" and step.seatbelt_events[-1].detail in sent["messages"][-1]["content"]
    assert {tool["function"]["name"] for tool in sent["tools"]} == {tool.name for tool in tools_for(["helpdesk"])}
    assert [event.kind for event in step.escalation.seatbelt_events] == ["invalid_call", "self_check"]
    assert step.escalation.calls[0].status == 0  # blocked by Lightning's own seatbelt, never run

    # Every executed call ran once, in order: the ticket was opened once, and nothing was replayed.
    made = [(call.tool, call.args) for s in result.steps for call in s.step.all_calls if call.status != 0]
    assert [(call.tool, call.args) for call in env.calls] == made
    assert [tool for tool, _ in made].count("create_ticket") == 1
    before = {ticket.id for ticket in Env("incident_prod_payments").state.tickets}
    assert {ticket.id for ticket in env.state.tickets} - before == {"HD-1042"}
    assert "update_ticket" not in {call.tool for call in env.calls}  # skipped after the retry limit: never executed

    # Ops gets the whole step's results: the ticket from the specialist's part, the look-up from Lightning's.
    instruction = ops.requests[0]["messages"][1]["content"]
    assert '"id":"HD-1042"' in instruction and '"tool":"get_ticket"' in instruction
    assert "Step 1 (helpdesk) answered: HD-1042 is open for the payments incident." in instruction
    assert result.answer == "HD-1042 is open for the payments incident.\n\nRolled payments back to revision 8."
    assert check(CASES["pr-multi-01"], env.state, result.answer).passed
    assert result.prompt_tokens == result.plan.prompt_tokens + sum(
        s.step.prompt_tokens + (s.step.escalation.prompt_tokens if s.step.escalation else 0) for s in result.steps)


def test_a_step_lightning_also_gives_up_on_fails_the_attempt() -> None:
    helpdesk = ScriptedModel(turn(*[("drop_tickets", {})] * 3))
    lightning = ScriptedModel(turn(*[("close_all", {})] * 3))
    ops = ScriptedModel()
    result, env = run("pr-multi-01", ScriptedModel(plan(*DEMO_PLAN)), escalation=lightning, helpdesk=helpdesk, ops=ops)
    step = result.steps[0].step
    assert step.escalated and step.escalation.gave_up and step.gave_up
    assert result.failed and result.order == [1] and not ops.requests and env.calls == []
    assert "Step 1 (helpdesk) failed, also after escalation to Lightning, so the team stopped there." in result.answer


def test_escalation_takes_only_steps_the_seatbelt_gave_up_on() -> None:
    lightning = ScriptedModel()
    helpdesk = ScriptedModel(turn(("create_ticket", TICKET)), *done("Opened HD-1042."))
    result, env = run("pr-multi-01", ScriptedModel(plan(*DEMO_PLAN)), escalation=lightning, helpdesk=helpdesk,
                      ops=ops_rolls_back())
    assert not lightning.requests and not any(s.step.escalated for s in result.steps)
    assert check(CASES["pr-multi-01"], env.state, result.answer).passed


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


def test_team_setups_differ_only_in_their_models() -> None:
    assert setup_models("team_base") == ("lightning", {"helpdesk": "student", "ops": "student"})
    assert setup_models("team_trained") == ("lightning", {"helpdesk": "helpdesk_r1", "ops": "ops_r1"})
    assert setup_models("team_small_planner") == ("student", {"helpdesk": "helpdesk_r1", "ops": "ops_r1"})
    assert setup_models("team_escalate") == setup_models("team_trained")
    # Only team_escalate escalates (hard rule 7), to Lightning.
    assert {setup: escalation_model(setup) for setup in team_setups()} == {
        "team_base": None, "team_escalate": "lightning", "team_small_planner": None, "team_trained": None}
