"""One task's environment: a fresh copy of its fixture, the tools of its APIs, and chaos.

Used by the Gym resources server (gym_env/server.py), by the verifier to run gold calls, and
directly by tests and any driver that runs outside Gym.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from gym_env import helpdesk, ops
from gym_env.chaos import Chaos, ChaosConfig, unavailable
from gym_env.state import load_fixture
from gym_env.tools import Tool, ToolError

TOOLS: dict[str, Tool] = {tool.name: tool for tool in (*helpdesk.TOOLS, *ops.TOOLS)}
APIS = ("helpdesk", "ops")


def tools_for(apis: Iterable[str]) -> list[Tool]:
    apis = set(apis)
    if unknown := apis - set(APIS):
        raise ValueError(f"no tools for apis {sorted(unknown)}")
    return [tool for tool in TOOLS.values() if tool.api in apis]


def tool_schemas(apis: Iterable[str]) -> list[dict[str, Any]]:
    return [tool.schema() for tool in tools_for(apis)]


@dataclass
class CallResult:
    tool: str
    args: Any
    status: int  # HTTP-style: 200 ok, 400/404 tool error, 503 transient
    output: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == 200


class Env:
    def __init__(
        self,
        fixture: str,
        apis: Iterable[str] = APIS,
        chaos: ChaosConfig | None = None,
        chaos_key: str = "",
    ) -> None:
        self.state = load_fixture(fixture)
        self.tools = {tool.name: tool for tool in tools_for(apis)}
        self.chaos = Chaos(chaos or ChaosConfig(), chaos_key)
        self.calls: list[CallResult] = []

    def call(self, name: str, args: Any) -> CallResult:
        """Run one tool call. Errors come back as results; a failed call never changes state."""
        try:
            output = self._run(name, args)
            result = CallResult(name, args, 200, output)
        except ToolError as error:
            result = CallResult(name, args, error.status, error.to_dict())
        self.calls.append(result)
        return result

    def _run(self, name: str, args: Any) -> dict[str, Any]:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(
                "unknown_tool",
                f"Unknown tool {name!r}. Available tools: {', '.join(self.tools)}.",
                status=404,
            )
        if self.chaos.strike():
            raise unavailable(tool.api)
        parsed = tool.validate(args)
        before = self.state.model_copy(deep=True)
        try:
            return tool.run(self.state, parsed)
        except ToolError:
            self.state = before  # tools check before they write; this makes it a guarantee
            raise
