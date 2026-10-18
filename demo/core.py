"""The demo's engine (task 5.3), without the UI: demo/app.py is the Gradio page on top of it, and demo/rehearse.py walks
the demo script with it.

run() sends one request through the team and, side by side, through single_base, each against its own fresh copy of
the demo's fixture (the payments incident of practice case pr-multi-01; its user, E1012, is the acting user). The team
is team_escalate: Lightning plans, the r1 adapters act, and Lightning finishes any step the seatbelt gives up on.
single_base is the student base model with every tool, the seatbelt on. Settings come from configs/eval.yaml, as in
the evals. Models are called through common/llm.py, so `mode` is LLM_MODE: live, record or replay. Each scenario
(active specialists, chaos rate) records under its own LLM_CACHE_TAG, so runs of the same request with and without
chaos don't overwrite each other's recordings. Chaos (gym_env/chaos.py) makes a share of tool calls return 503,
seeded by the case, so a replay repeats it.

Each side comes back as markdown panels: the plan (team), every step with its calls, seatbelt events and any
escalation, what changed in the environment, the answer, the verifier's verdict (only for the case's own request),
and the counters: steps handled by small models, and the request's cost at configs/eval.yaml's prices.
"""

import asyncio
import json
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.orchestrator import Escalation, Member, TeamResult, escalation_model, load_practice_case, run_team, setup_models
from agents.specialist import load_specialists
from common.eval_config import REPO, load_eval_config
from common.llm import ReplayMiss
from common.schemas import Case, Step
from common.tool_loop import ModelFn, StepResult, chat_tools, env_tools, llm_model, run_step
from eval.run_eval import cost_usd
from gym_env.chaos import ChaosConfig
from gym_env.env import Env, tool_schemas
from gym_env.tasks import system_prompt
from gym_env.verify import check

DEMO_CASE = "pr-multi-01"
TEAM_SETUP = "team_escalate"
SINGLE_MODEL = "student"  # single_base
# Chaos draws follow the calls' order (gym_env/chaos.py). With seed 0 the demo's first five calls all pass at rate 0.5;
# with seed 5 at rate 0.3 calls 1, 4, 7 and 12 fail, so the automatic retries show and none runs out.
CHAOS_RATE, CHAOS_SEED = 0.3, 5
SPECIALISTS = {"helpdesk": ("🎫", "Helpdesk"), "ops": ("🛠️", "Ops")}  # icon and name of each specialist
RUNS = REPO / "runs"
RESULT_TABLES = {"ops ladder": RUNS / "practice_ladder_r1.md", "system": RUNS / "practice_system.md",
                 "vLLM serving": RUNS / "bench" / "summary.md"}

ModelFactory = Callable[[str], ModelFn]


@dataclass
class Side:
    """One system's run of the request, as panels."""

    title: str
    detail: str = ""  # markdown: the plan and steps
    state: str = ""  # markdown: what changed in the environment
    answer: str = ""
    verdict: str = ""  # empty when the request isn't the case's own
    counters: str = ""
    error: str | None = None
    numbers: dict[str, Any] = field(default_factory=dict)  # for demo/rehearse.py


def demo_case() -> Case:
    return load_practice_case(DEMO_CASE)


def cache_tag(active: list[str], chaos_rate: float, chaos_seed: int = CHAOS_SEED) -> str:
    return f"demo:{'+'.join(sorted(active)) or 'none'}:chaos={chaos_rate:g}:seed={chaos_seed}"


def fresh_env(case: Case, chaos_rate: float, chaos_seed: int = CHAOS_SEED) -> Env:
    chaos = ChaosConfig(enabled=chaos_rate > 0, rate=chaos_rate, seed=chaos_seed)
    return Env(case.fixture, apis=case.apis, chaos=chaos, chaos_key=case.id)


def label(agent: str) -> str:
    icon, name = SPECIALISTS.get(agent, ("🧰", agent))
    return f"{icon} {name}"


def call_lines(step: Step) -> list[str]:
    """The step's calls and seatbelt events in the order they happened: by turn, each event after its turn's calls."""
    lines = []
    for call in step.calls:
        output = json.dumps(call.output)
        lines.append((call.turn, 0, f"- `{call.tool}` `{json.dumps(call.args)}` → {call.status or 'blocked by the seatbelt'}: "
                                    f"`{output[:180]}{'…' if len(output) > 180 else ''}`"))
    for event in step.seatbelt_events:
        lines.append((event.turn, 1, f"- 🛡️ seatbelt {event.kind.replace('_', ' ')}: {event.detail}"))
    return [text for _, _, text in sorted(lines, key=lambda line: line[:2])]


def step_panel(number: int, result: StepResult) -> str:
    step = result.step
    lines = [f"**Step {number}: {label(step.agent)}** on `{step.model}`", *call_lines(step)]
    if step.escalation is not None:
        lines += [f"**⚡ Lightning took the step over** on `{step.escalation.model}`", *call_lines(step.escalation)]
    lines.append(f"> {result.answer or '(no answer)'}")
    return "\n".join(lines)


def plan_panel(result: TeamResult) -> str:
    plan = result.plan
    lines = [f"**🧭 Lightning's plan** (`{plan.model}`)"]
    if plan.error:
        lines.append(f"- ❌ {plan.error}")
    lines.append(f"- *{plan.analysis}*")
    for number, step in enumerate(plan.steps, start=1):
        after = f" (after {', '.join(map(str, step.depends_on))})" if step.depends_on else ""
        lines.append(f"{number}. {label(step.agent)}{after}: {step.instruction}")
    if plan.uncovered:
        lines.append(f"- Not covered by the active specialists: {'; '.join(plan.uncovered)}")
    if plan.reply:
        lines.append(f"- Reply: {plan.reply}")
    return "\n".join(lines)


def pods(case: Case, env: Env, deployments: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Each deployment's pods in `env`'s state, looked up on a copy without chaos, so looking is no part of the run."""
    view = fresh_env(case, 0.0)
    view.state = env.state.model_copy(deep=True)
    out = {}
    for namespace, name in deployments:
        found = view.call("get_pods", {"namespace": namespace, "deployment": name}).output.get("pods", [])
        out[(namespace, name)] = ", ".join(f"{n} {status}" for status, n in Counter(p["status"] for p in found).items())
    return out


def state_panel(case: Case, env: Env) -> str:
    """What the run changed: tickets opened or updated, deployments rolled back or scaled, and their pods before and
    after."""
    before = fresh_env(case, 0.0)
    old = {ticket.id: ticket for ticket in before.state.tickets}
    lines = []
    for ticket in env.state.tickets:
        if ticket.id not in old:
            lines.append(f"- Opened **{ticket.id}**: {ticket.priority} {ticket.category} for {ticket.employee_id}, "
                         f"“{ticket.summary}”" + "".join(f"; note: “{note}”" for note in ticket.notes))
        elif ticket != old[ticket.id]:
            was = old[ticket.id]
            changed = [f"{name} {getattr(was, name)} → {getattr(ticket, name)}" for name in ("status", "priority")
                       if getattr(was, name) != getattr(ticket, name)]
            if len(ticket.notes) > len(was.notes):
                changed.append(f"{len(ticket.notes) - len(was.notes)} note(s) added")
            lines.append(f"- Updated **{ticket.id}**: {', '.join(changed) or 'changed'}")
    for change in env.state.changes:
        ticket = f" with change ticket {change.change_ticket}" if change.change_ticket else ""
        unit = "revision" if change.action == "rollback" else "replicas"
        lines.append(f"- {change.action.capitalize()} **{change.namespace}/{change.deployment}**: {unit} "
                     f"{change.before} → {change.after}{ticket}")
    deployments = list(dict.fromkeys((c.namespace, c.deployment) for c in env.state.changes))
    was = pods(case, before, deployments)
    for key, now in pods(case, env, deployments).items():
        lines.append(f"- Pods of {key[0]}/{key[1]}: {was[key]} → **{now}**")
    return "\n".join(lines) or "- Nothing changed."


def verdict_line(case: Case, request: str, env: Env, answer: str) -> str:
    if request.strip() != case.request:
        return ""
    verdict = check(case, env.state, answer)
    if verdict.passed:
        return "✅ The verifier passes it (the state matches the gold calls' and the answer has what it needs)."
    problems = verdict.state_diff[:3] + [f"answer lacks {t!r}" for t in verdict.missing_from_answer]
    return "❌ The verifier fails it: " + "; ".join(problems)


def retries(steps: list[Step]) -> int:
    parts = [part for step in steps for part in (step, step.escalation) if part is not None]
    return sum(event.kind == "transient_retry" for part in parts for event in part.seatbelt_events)


async def team_side(case: Case, request: str, active: list[str], chaos: tuple[float, int], models: ModelFactory) -> Side:
    config = load_eval_config()
    planner, specialist_models = setup_models(TEAM_SETUP)
    target = escalation_model(TEAM_SETUP)
    specialists = load_specialists()
    members = [Member(specialists[name], models(specialist_models[name]), specialist_models[name]) for name in active]
    env = fresh_env(case, *chaos)
    result = await run_team(request, system_prompt(case), chat_tools(tool_schemas(case.apis)), models(planner), planner,
                            members, env_tools(env), max_steps=config.max_steps, seatbelt=config.seatbelt,
                            escalation=Escalation(models(target), target) if target else None)
    steps = [r.step for r in result.steps]
    cost = cost_usd({"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens}, steps,
                    result.plan, config)
    small = sum(not s.escalated for s in steps)
    side = Side(title=f"🧭 The team ({', '.join(label(a) for a in active) or 'no specialists'})")
    side.detail = "\n\n".join([plan_panel(result), *(step_panel(n, r) for n, r in zip(result.order, result.steps))])
    side.state, side.answer = state_panel(case, env), result.answer
    side.verdict = verdict_line(case, request, env, result.answer)
    side.counters = (f"**{small}/{len(steps)}** steps handled by small models · **${cost:.4f}** of GPU time for this request "
                     f"(${1000 * cost:.2f} per 1,000) · {retries(steps)} automatic 503 retries · {result.latency_s:.1f} s")
    side.numbers = {"steps": len(steps), "small_model_steps": small, "escalated": len(steps) - small,
                    "plan_steps": [s.agent for s in result.plan.steps], "uncovered": result.plan.uncovered,
                    "calls": [c.tool for s in steps for c in s.all_calls],
                    "retries": retries(steps), "cost_usd": cost, "passed": side.verdict.startswith("✅"),
                    "answer": result.answer}
    return side


async def single_side(case: Case, request: str, chaos: tuple[float, int], models: ModelFactory) -> Side:
    config = load_eval_config()
    env = fresh_env(case, *chaos)
    messages = [{"role": "system", "content": system_prompt(case)}, {"role": "user", "content": request}]
    result = await run_step(messages, chat_tools(tool_schemas(case.apis)), models(SINGLE_MODEL), env_tools(env),
                            max_steps=config.max_steps, seatbelt=config.seatbelt, agent="single", model_name=SINGLE_MODEL)
    cost = cost_usd({"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens},
                    [result.step], None, config)
    side = Side(title="🧰 single_base: one small model with every tool")
    side.detail = "\n".join([f"**{SINGLE_MODEL}**, all {len(tool_schemas(case.apis))} tools", *call_lines(result.step),
                             f"> {result.answer or '(no answer)'}"])
    side.state, side.answer = state_panel(case, env), result.answer
    side.verdict = verdict_line(case, request, env, result.answer)
    side.counters = (f"**1/1** steps by a small model · **${cost:.4f}** of GPU time for this request (${1000 * cost:.2f} per 1,000) · "
                     f"{retries([result.step])} automatic 503 retries · {result.latency_s:.1f} s")
    side.numbers = {"calls": [c.tool for c in result.step.calls], "retries": retries([result.step]), "cost_usd": cost,
                    "passed": side.verdict.startswith("✅"), "gave_up": result.gave_up, "answer": result.answer}
    return side


async def guarded(title: str, run: Any) -> Side:
    """A side whose models weren't recorded (replay) or whose server is down comes back as an error panel."""
    try:
        return await run
    except ReplayMiss as error:
        return Side(title=title, error=f"Not recorded for replay ({error}). Run it live (or rehearse) first.")
    except Exception as error:  # a server that's down, for example: the demo shows it instead of crashing
        return Side(title=title, error=f"{type(error).__name__}: {error}")


def run(request: str, active: list[str], chaos_rate: float = 0.0, mode: str = "live",
        models: ModelFactory | None = None, chaos_seed: int = CHAOS_SEED) -> tuple[Side, Side]:
    """The team and single_base on one request. `mode` is LLM_MODE for this run (live, record or replay); a chaos rate
    of 0 is chaos off."""
    case = demo_case()
    if models is None:
        config = load_eval_config()

        def models(name: str) -> ModelFn:
            return llm_model(name, config.sampling(), config.thinking)

    saved = {key: os.environ.get(key) for key in ("LLM_MODE", "LLM_CACHE_TAG")}
    os.environ["LLM_MODE"], os.environ["LLM_CACHE_TAG"] = mode, cache_tag(active, chaos_rate, chaos_seed)
    chaos = (chaos_rate, chaos_seed)
    try:
        async def both() -> tuple[Side, Side]:
            return await asyncio.gather(
                guarded("🧭 The team", team_side(case, request, active, chaos, models)),
                guarded("🧰 single_base", single_side(case, request, chaos, models)),
            )

        return asyncio.run(both())
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def result_tables() -> dict[str, str]:
    """The results to show next to the live run, as the scripts that made them wrote them."""
    return {name: path.read_text() if path.exists() else f"*{path.relative_to(REPO)} hasn't been made yet.*"
            for name, path in RESULT_TABLES.items()}


def ladder_rows(path: Path = RESULT_TABLES["ops ladder"]) -> list[dict[str, Any]]:
    """The specialist ladder's rows (eval/report.py's table): setup, API and cases passed."""
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = dict(zip(header, (cell.strip() for cell in line.strip("|").split("|"))))
        passed, total = cells["cases passed"].split()[0].split("/")
        rows.append({"setup": cells["setup"], "api": cells["api"], "cases passed": int(passed), "cases": int(total)})
    return rows
