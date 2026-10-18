"""A specialist: one model or adapter with one API's tools, run through the tool loop (seatbelt included).

agents/registry.json lists every specialist with its API and card (agents/cards/<name>.json, with the A2A
Agent Card fields name, description and skills), and which ones are active. The orchestrator plans from the
active cards (agents/orchestrator.py) and hands each specialist one step at a time.

A step's conversation is the single-model setups' system prompt (acting user and time, gym_env/tasks.py) and
the step's instruction, with the results of the steps it depends on, as the user turn. The specialist never
sees the whole request, so it isn't asked to do another specialist's part. It gets only its own API's tools: a
call to any other tool comes back as unknown (from the seatbelt, or from here when the seatbelt is off).

finish() is escalation (team_escalate only, CLAUDE.md "Escalation"): Lightning takes over a step the seatbelt gave
up on, in the same conversation, with the same tools and seatbelt settings.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.eval_config import REPO, SeatbeltConfig
from common.schemas import Step
from common.seatbelt import error_output
from common.tool_loop import ModelFn, StepResult, ToolFn, run_step
from gym_env.env import tools_for

REGISTRY = REPO / "agents" / "registry.json"
# The one message Lightning gets on top of the step's conversation when it takes the step over.
HANDOFF = (
    "Another assistant worked on this step until the seatbelt stopped it ({reason}). Take over and finish the step "
    "from here. The tool calls above that returned results have already run, so don't repeat one that succeeded; "
    "a call marked as not executed never ran."
)


@dataclass(frozen=True)
class Specialist:
    name: str
    api: str
    card: dict[str, Any]

    def tools(self, chat_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """This specialist's tools among a request's chat-format tool definitions."""
        names = {tool.name for tool in tools_for([self.api])}
        return [tool for tool in chat_tools if tool["function"]["name"] in names]


def load_specialists(path: Path = REGISTRY) -> dict[str, Specialist]:
    """Every specialist in the registry, by name."""
    specialists = {}
    for name, entry in json.loads(path.read_text())["specialists"].items():
        card = json.loads((path.parent / entry["card"]).read_text())
        if card["name"] != name:
            raise ValueError(f"{entry['card']} is named {card['name']!r}, but the registry lists it as {name!r}")
        specialists[name] = Specialist(name, entry["api"], card)
    return specialists


def active_names(path: Path = REGISTRY) -> list[str]:
    active = json.loads(path.read_text())["active"]
    if unknown := set(active) - set(load_specialists(path)):
        raise ValueError(f"{path.name} activates specialists it doesn't list: {sorted(unknown)}")
    return active


def active_specialists(path: Path = REGISTRY) -> list[Specialist]:
    specialists = load_specialists(path)
    return [specialists[name] for name in active_names(path)]


def inactive_names(path: Path = REGISTRY) -> list[str]:
    """Specialists in the registry that are switched off (the demo toggles them; evals need them all)."""
    active = set(active_names(path))
    return [name for name in load_specialists(path) if name not in active]


def own_tools(specialist: Specialist, tools: list[dict[str, Any]], call_tool: ToolFn) -> tuple[list[dict[str, Any]], ToolFn]:
    """The specialist's own tools among a request's chat-format tools, and a ToolFn that refuses every other tool."""
    own = specialist.tools(tools)
    names = [tool["function"]["name"] for tool in own]

    async def call_own(name: str, args: Any) -> tuple[int, Any]:
        if name not in names:
            return 404, error_output("unknown_tool", f"Unknown tool {name!r}. Available tools: {', '.join(names)}.")
        return await call_tool(name, args)

    return own, call_own


async def run(
    specialist: Specialist,
    system_prompt: str,
    instruction: str,
    tools: list[dict[str, Any]],
    model: ModelFn,
    call_tool: ToolFn,
    *,
    max_steps: int,
    seatbelt: SeatbeltConfig | None,
    model_name: str,
) -> StepResult:
    """Run one step. `tools` are the request's chat-format tools; the specialist gets its own API's."""
    own, call_own = own_tools(specialist, tools, call_tool)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]
    return await run_step(messages, own, model, call_own, max_steps=max_steps, seatbelt=seatbelt,
                          agent=specialist.name, model_name=model_name)


async def finish(
    specialist: Specialist,
    failed: StepResult,
    tools: list[dict[str, Any]],
    model: ModelFn,
    call_tool: ToolFn,
    *,
    max_steps: int,
    seatbelt: SeatbeltConfig | None,
    model_name: str,
) -> StepResult:
    """Escalation: `model` (Lightning) finishes a step the seatbelt gave up on. It gets the step's conversation so far,
    with every call already executed and its result, plus HANDOFF, and works on with the specialist's tools and the
    same seatbelt settings: its own retries and up to `max_steps` turns, since the specialist used up its own. It
    continues the step and never restarts it, so no executed call runs again. The result is the whole step: the
    specialist's part, with Lightning's as its `escalation`; the whole conversation; and Lightning's answer."""
    reason = next((event.detail for event in reversed(failed.step.seatbelt_events) if event.kind == "gave_up"),
                  "the step failed")
    own, call_own = own_tools(specialist, tools, call_tool)
    messages = [*failed.messages, {"role": "user", "content": HANDOFF.format(reason=reason)}]
    lightning = await run_step(messages, own, model, call_own, max_steps=max_steps, seatbelt=seatbelt,
                               agent=specialist.name, model_name=model_name)
    step = Step(**{**dict(failed.step), "escalated": True, "escalation": lightning.step})
    return StepResult(step, lightning.messages, lightning.answer, failed.prompt_tokens + lightning.prompt_tokens,
                      failed.completion_tokens + lightning.completion_tokens, failed.latency_s + lightning.latency_s)
