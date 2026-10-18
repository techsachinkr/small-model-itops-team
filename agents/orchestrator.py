"""Lightning orchestrator: plans steps from the specialist cards and runs them in dependency order.

plan(): one structured-output call to the planner (Lightning in team_base and team_trained). It sees the
single setups' system prompt (acting user and time), the active specialists' cards and the request, and
answers with plan_schema(): `analysis` first, where it lists the actions the user asked for, decides whether to
carry them out, ask or decline, and says where each detail comes from, before writing any step (structured
answers run with thinking off, common/tool_loop.py); at most MAX_PLAN_STEPS `steps`, each
{agent, instruction, depends_on} with `agent` one of the active specialists; `uncovered`, the parts of the
request no active specialist covers; and `reply`, its own message when no step should run (a question for
missing information, or a decline). The plan goes into the trace (common.schemas.Plan).

run_team(): runs the steps in dependency order, keeping the plan's order where the dependencies allow. A
step's instruction gets the answer and successful tool results of each step it depends on, so an ID that one
specialist finds or creates (like a new ticket's) reaches the next. The team stops at the first failed step:
a failed step fails the case (CLAUDE.md "Seatbelt"; escalation is task 3.3). The final answer joins the
planner's reply, each step's answer and a sentence naming the uncovered parts.

Like common/tool_loop.py, it needs only async functions: a ModelFn for the planner and one per specialist,
and a ToolFn. So the same code runs inside NeMo Gym (ItopsTeamAgent in common/gym_agent.py) and outside it,
as in this CLI on common/llm.py and a local Env. The CLI runs one practice case and prints the plan, every
step and the verifier's verdict; with LLM_MODE=record it also saves every response for replay:

    uv run python -m agents.orchestrator --case pr-multi-01 [--setup team_base] [--active helpdesk]
"""

import argparse
import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import ValidationError

from agents.specialist import Specialist, active_names, load_specialists
from agents.specialist import run as run_specialist
from common.eval_config import EVAL_CONFIG, REPO, SeatbeltConfig, load_eval_config
from common.schemas import Plan, PlanStep, Strict
from common.tool_loop import ModelFn, StepResult, ToolFn

if TYPE_CHECKING:
    from common.schemas import Case
    from gym_env.env import Env
    from gym_env.verify import Verdict

GYM_CONFIGS = REPO / "gym_configs"
TEAM_CONFIG = GYM_CONFIGS / "eval_team.yaml"  # shared by the team setups; gym_configs/team_*.yaml pick the models
MAX_PLAN_STEPS = 6

# Revised after the first practice runs (task 3.1, runs/team_*_practice): Lightning planned requests it should have
# declined or asked about, opened change tickets nobody asked for (the first prompt's example said the ticket had to
# be "found or opened"), and wrote instructions without the acting user, with a category other than the user's, as
# questions, or as steps that run only "if" something happens.
PLANNER_PROMPT = """You plan how a team of IT specialists carries out a user's request. You don't act yourself: each step of your plan goes to one specialist, who carries it out with its own tools.

The request comes with this context, which every specialist also sees:
{context}

The specialists (their cards):
{cards}

Answer with a JSON object:
- analysis: first, the actions the user asked for, in the user's words, and nothing more. Then whether to carry them out, ask the user or decline (see the rules below). Then which specialist does each action, and where each detail it needs comes from: the request, the context or an earlier step's result.
- steps: at most {max_steps}, in the order to run them. Each has `agent` (a specialist's name), `instruction` and `depends_on` (the numbers of the earlier steps whose results it needs, counting from 1).
- uncovered: each part of the request that none of these specialists can do at all. Leave those parts out of the steps. Empty if every part is covered. A detail that a step will look up is never uncovered.
- reply: empty when there are steps. When there are none, your message to the user.

Rules:
- Plan only the actions the user asked for: no extra checks or follow-ups, and no steps that should run only if something turns out a certain way. Open a ticket only when the user asks for one.
- A change in prod needs the ID of an open helpdesk ticket as its change ticket: the one the user gives or points to (a helpdesk step looks it up), or the one a step opens because the user asked for a new ticket. If the user did neither, don't plan the change: ask for a change ticket in `reply`.
- Decline, with no steps and the reason in `reply`, a request that would cause an outage in prod or undo work that isn't the user's. Ask, with no steps, only for information that no specialist could look up with its tools.
- Give each specialist one step per action: it looks up what it needs itself (an ops step finds a deployment's rollout history before rolling it back). Add a separate step only when another specialist needs its result first.
- A specialist sees only its own instruction and the results of the steps it depends on, never the user's request. Write each instruction as a request from the user to that specialist, never as a question: name the employee (the current user's ID from the context, when it's for them), and keep every detail the user gave (category, priority, namespace, names, wording) exactly, without listing alternatives or copying fields from the cards.
- For a value an earlier step will find or create, like a ticket ID or the change a ticket asks for, tell the specialist to take it from that step's result rather than guessing it, and to change nothing if it isn't there."""


class PlanAnswer(Strict):
    """The planner's answer (plan_schema)."""

    analysis: str
    steps: list[PlanStep]
    uncovered: list[str]
    reply: str


def plan_schema(agents: list[str]) -> dict[str, Any]:
    """The JSON schema the planner answers with; `agents` are the active specialists' names."""
    step = {
        "type": "object",
        "properties": {
            "agent": {"type": "string", "enum": agents} if agents else {"type": "string"},
            "instruction": {"type": "string"},
            "depends_on": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        },
        "required": ["agent", "instruction", "depends_on"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "steps": {"type": "array", "items": step, "maxItems": MAX_PLAN_STEPS if agents else 0},
            "uncovered": {"type": "array", "items": {"type": "string"}},
            "reply": {"type": "string"},
        },
        "required": ["analysis", "steps", "uncovered", "reply"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class Member:
    """An active specialist with the model it runs on in this setup."""

    specialist: Specialist
    model: ModelFn
    model_name: str


@dataclass
class TeamResult:
    plan: Plan
    steps: list[StepResult]  # in the order they ran
    order: list[int]  # the plan step number of each entry in `steps`
    answer: str
    latency_s: float  # wall time of the plan and every step
    notes: list[str] = field(default_factory=list)  # why the team stopped early, if it did

    @property
    def failed(self) -> bool:
        """No valid plan, or a failed step: the attempt fails (scores 0 in Gym)."""
        return self.plan.error is not None or any(result.gave_up for result in self.steps)

    @property
    def prompt_tokens(self) -> int:
        return self.plan.prompt_tokens + sum(result.prompt_tokens for result in self.steps)

    @property
    def completion_tokens(self) -> int:
        return self.plan.completion_tokens + sum(result.completion_tokens for result in self.steps)


async def plan(request: str, system_prompt: str, cards: list[dict[str, Any]], planner: ModelFn, planner_model: str) -> Plan:
    names = [card["name"] for card in cards]
    prompt = PLANNER_PROMPT.format(context=system_prompt, cards=json.dumps(cards, indent=1), max_steps=MAX_PLAN_STEPS)
    reply = await planner([{"role": "system", "content": prompt}, {"role": "user", "content": request}], None,
                          plan_schema(names))
    tokens = {"model": planner_model, "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens}
    try:
        answer = PlanAnswer.model_validate_json(reply.text)
    except ValidationError:
        return Plan(analysis="", steps=[], uncovered=[], reply="",
                    error=f"the planner's answer isn't a valid plan: {reply.text[:300]!r}", **tokens)
    if unknown := sorted({step.agent for step in answer.steps} - set(names)):
        return Plan(**answer.model_dump(), error=f"the plan uses specialists that aren't active: {unknown}", **tokens)
    return Plan(**answer.model_dump(), **tokens)


def run_order(steps: list[PlanStep]) -> list[int]:
    """0-based step indexes in dependency order, keeping the plan's order where the dependencies allow.
    Dependencies on steps that don't exist are ignored; steps in a cycle run in plan order."""
    needs = [{d - 1 for d in step.depends_on if 0 < d <= len(steps) and d - 1 != i} for i, step in enumerate(steps)]
    order: list[int] = []
    while len(order) < len(steps):
        waiting = [i for i in range(len(steps)) if i not in order]
        ready = [i for i in waiting if needs[i] <= set(order)]
        order.append((ready or waiting)[0])
    return order


def step_instruction(number: int, plan: Plan, results: dict[int, StepResult]) -> str:
    """Plan step `number`'s instruction, with the answer and successful tool results of the steps it depends on."""
    step = plan.steps[number - 1]
    needed = [d for d in dict.fromkeys(step.depends_on) if d in results]
    if not needed:
        return step.instruction
    lines = [step.instruction, "", "Results of the earlier steps this one depends on:"]
    for d in needed:
        result = results[d]
        lines.append(f"Step {d} ({plan.steps[d - 1].agent}) answered: {result.answer or '(no answer)'}")
        outputs = [{"tool": call.tool, "output": call.output} for call in result.step.calls if call.status == 200]
        if outputs:
            lines.append(f"Its tool results: {json.dumps(outputs, separators=(',', ':'))}")
    return "\n".join(lines)


def final_answer(plan: Plan, steps: list[StepResult], notes: list[str]) -> str:
    parts = [plan.reply.strip()] if plan.reply.strip() else []
    parts += [result.answer.strip() for result in steps if result.answer.strip()]
    if plan.uncovered:
        missing = "; ".join(part.strip().rstrip(".") for part in plan.uncovered)
        parts.append(f"Not done, because no specialist on the team covers it: {missing}.")
    return "\n\n".join(parts + notes)


async def run_team(
    request: str,
    system_prompt: str,
    tools: list[dict[str, Any]],
    planner: ModelFn,
    planner_model: str,
    members: list[Member],
    call_tool: ToolFn,
    *,
    max_steps: int,
    seatbelt: SeatbeltConfig | None,
) -> TeamResult:
    """Plan the request with the members' cards, then run the plan. `tools` are the request's chat-format
    tools (every API's); `max_steps` and `seatbelt` apply to each step, as in the single setups."""
    start = time.perf_counter()
    by_name = {member.specialist.name: member for member in members}
    the_plan = await plan(request, system_prompt, [member.specialist.card for member in members], planner, planner_model)
    results: dict[int, StepResult] = {}
    order: list[int] = []
    notes: list[str] = []
    if the_plan.error is not None:
        notes.append("The request couldn't be planned, so nothing was done.")
    else:
        steps = the_plan.steps[:MAX_PLAN_STEPS]
        if len(the_plan.steps) > MAX_PLAN_STEPS:
            notes.append(f"The plan had {len(the_plan.steps)} steps; only the first {MAX_PLAN_STEPS} ran.")
        for index in run_order(steps):
            number = index + 1
            member = by_name[steps[index].agent]
            result = await run_specialist(
                member.specialist, system_prompt, step_instruction(number, the_plan, results), tools, member.model,
                call_tool, max_steps=max_steps, seatbelt=seatbelt, model_name=member.model_name,
            )
            results[number] = result
            order.append(number)
            if result.gave_up:
                notes.append(f"Step {number} ({member.specialist.name}) failed, so the team stopped there.")
                break
    ran = [results[number] for number in order]
    return TeamResult(the_plan, ran, order, final_answer(the_plan, ran, notes), time.perf_counter() - start, notes)


def failure_label(plan: Plan | None, gold_apis: set[str], specialists: dict[str, Specialist]) -> Literal["routing", "specialist"]:
    """How a failed team attempt is labeled (CLAUDE.md "Scoring"): routing if the plan's agents don't match the
    APIs in the case's gold calls (or there was no valid plan), otherwise specialist."""
    if plan is None or plan.error is not None:
        return "routing"
    return "routing" if {specialists[agent].api for agent in plan.agents} != gold_apis else "specialist"


def team_setups() -> list[str]:
    return sorted(path.stem for path in GYM_CONFIGS.glob("team_*.yaml"))


def setup_models(setup: str) -> tuple[str, dict[str, str]]:
    """A team setup's planner model and each specialist's model, from gym_configs/eval_team.yaml and the setup's
    own file (the same values Gym gets)."""
    values = yaml.safe_load(TEAM_CONFIG.read_text()) | yaml.safe_load((GYM_CONFIGS / f"{setup}.yaml").read_text())
    return values["planner_model_name"], {name: values[f"{name}_model_name"] for name in load_specialists()}


def describe(result: TeamResult) -> list[str]:
    """The plan and every step, for people reading a run."""
    plan = result.plan
    lines = [f"Plan ({plan.model}, {plan.prompt_tokens} prompt + {plan.completion_tokens} completion tokens)"]
    if plan.error:
        lines.append(f"  ERROR: {plan.error}")
    lines.append(f"  analysis: {plan.analysis}")
    for number, step in enumerate(plan.steps, start=1):
        after = f" (after {', '.join(map(str, step.depends_on))})" if step.depends_on else ""
        lines.append(f"  {number}. {step.agent}{after}: {step.instruction}")
    if plan.uncovered:
        lines.append(f"  uncovered: {plan.uncovered}")
    if plan.reply:
        lines.append(f"  reply: {plan.reply}")
    for number, result_step in zip(result.order, result.steps):
        step = result_step.step
        lines.append(f"Step {number}: {step.agent} on {step.model} ({result_step.prompt_tokens} prompt + "
                     f"{result_step.completion_tokens} completion tokens, {result_step.latency_s:.1f} s)")
        for call in step.calls:
            lines.append(f"  {call.tool} {json.dumps(call.args)} -> {call.status} {json.dumps(call.output)[:300]}")
        for event in step.seatbelt_events:
            lines.append(f"  seatbelt {event.kind}: {event.detail}")
        lines.append(f"  answer: {result_step.answer}")
    lines.append(f"Final answer ({result.latency_s:.1f} s in all):\n{result.answer}")
    return lines


def load_practice_case(case_id: str) -> "Case":
    """A case from data/practice/ (this CLI never reads data/test/, hard rule 4)."""
    from gym_env.tasks import read_cases

    for path in sorted((REPO / "data" / "practice").glob("*.jsonl")):
        for case in read_cases(path):
            if case.id == case_id:
                return case
    raise SystemExit(f"No practice case {case_id!r} in data/practice/.")


def verdict_lines(verdict: "Verdict", result: TeamResult) -> list[str]:
    lines = [f"Verifier: {'PASS' if verdict.passed else 'FAIL'} (state match {verdict.state_match}, "
             f"answer ok {verdict.answer_ok})"]
    lines += [f"  state: {line}" for line in verdict.state_diff]
    lines += [f"  missing from the answer: {text!r}" for text in verdict.missing_from_answer]
    lines += [f"  forbidden in the answer: {text!r}" for text in verdict.forbidden_in_answer]
    if result.failed:
        lines.append("  The team failed (no valid plan, or a step gave up), so in Gym this attempt scores 0.")
    return lines


def pods_after(env: "Env") -> list[str]:
    """The pods of every deployment the run changed, as they are now."""
    lines = []
    for namespace, name in dict.fromkeys((change.namespace, change.deployment) for change in env.state.changes):
        pods = env.call("get_pods", {"namespace": namespace, "deployment": name}).output["pods"]
        statuses = Counter(pod["status"] for pod in pods)
        lines.append(f"After the run, {namespace}/{name} has " + ", ".join(f"{n} {s}" for s, n in statuses.items())
                     + " pods.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the team on one practice case outside NeMo Gym and verify it.")
    parser.add_argument("--case", required=True, help="a case id from data/practice/*.jsonl")
    parser.add_argument("--setup", default="team_trained", choices=team_setups())
    parser.add_argument("--active", nargs="+", help="activate only these specialists (default: agents/registry.json)")
    parser.add_argument("--config", type=Path, default=EVAL_CONFIG, help="eval config (default configs/eval.yaml)")
    args = parser.parse_args()

    from common.tool_loop import chat_tools, env_tools, llm_model
    from gym_env.env import Env, tool_schemas
    from gym_env.tasks import system_prompt
    from gym_env.verify import check

    case = load_practice_case(args.case)
    config = load_eval_config(args.config)
    planner_model, models = setup_models(args.setup)
    specialists = load_specialists()
    names = args.active or active_names()
    if unknown := set(names) - set(specialists):
        raise SystemExit(f"Unknown specialists: {sorted(unknown)}; the registry lists {sorted(specialists)}.")

    def model(name: str) -> ModelFn:
        return llm_model(name, config.sampling(), config.thinking)

    members = [Member(specialists[name], model(models[name]), models[name]) for name in names]
    env = Env(case.fixture, apis=case.apis)  # chaos off
    result = asyncio.run(run_team(
        case.request, system_prompt(case), chat_tools(tool_schemas(case.apis)), model(planner_model), planner_model,
        members, env_tools(env), max_steps=config.max_steps, seatbelt=config.seatbelt,
    ))
    verdict = check(case, env.state, result.answer)
    print(f"{case.id} ({case.kind}, {case.fixture}, {case.user_id}) on {args.setup} with {', '.join(names)}:")
    print(f"Request: {case.request}")
    print("\n".join(describe(result) + verdict_lines(verdict, result) + pods_after(env)))


if __name__ == "__main__":
    main()
