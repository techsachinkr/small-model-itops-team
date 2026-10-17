"""A specialist: one model or adapter with one API's tools, run through the tool loop (seatbelt included).

agents/registry.json lists every specialist with its API and card (agents/cards/<name>.json, with the A2A
Agent Card fields name, description and skills), and which ones are active. The orchestrator plans from the
active cards (agents/orchestrator.py) and hands each specialist one step at a time.

A step's conversation is the single-model setups' system prompt (acting user and time, gym_env/tasks.py) and
the step's instruction, with the results of the steps it depends on, as the user turn. The specialist never
sees the whole request, so it isn't asked to do another specialist's part. It gets only its own API's tools: a
call to any other tool comes back as unknown (from the seatbelt, or from here when the seatbelt is off).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.eval_config import REPO, SeatbeltConfig
from common.seatbelt import error_output
from common.tool_loop import ModelFn, StepResult, ToolFn, run_step
from gym_env.env import tools_for

REGISTRY = REPO / "agents" / "registry.json"


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
    own = specialist.tools(tools)
    names = [tool["function"]["name"] for tool in own]

    async def own_tools(name: str, args: Any) -> tuple[int, Any]:
        if name not in names:
            return 404, error_output("unknown_tool", f"Unknown tool {name!r}. Available tools: {', '.join(names)}.")
        return await call_tool(name, args)

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}]
    return await run_step(messages, own, model, own_tools, max_steps=max_steps, seatbelt=seatbelt,
                          agent=specialist.name, model_name=model_name)
