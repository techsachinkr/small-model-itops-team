"""Runs a model through the seatbelt against the environment's tools.

The loop speaks OpenAI chat format and needs only two async functions, so the same loop runs inside
NeMo Gym and outside it:
- a ModelFn sends the conversation, plus the tools or a JSON schema (for the self-check), to a model
  and returns a ModelReply. A request with a JSON schema is answered with thinking off, so the schema
  constrains the whole output (with thinking on, the student sometimes never closed its reasoning and
  the answer came back empty). The Gym agent (common/gym_agent.py) calls Gym's model server;
  `llm_model` calls common/llm.py.
- a ToolFn runs one call and returns (status, output). The Gym agent calls the resources server;
  `env_tools` calls a gym_env Env directly.

One run_step() is one step: the model works until it answers without tool calls, or for at most
`max_steps` turns. With the seatbelt off (base_raw only), every call goes straight to the tools and
every result goes back to the model, as in Gym's simple agent. With it on, see common/seatbelt.py.
"""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from common import llm
from common.eval_config import SeatbeltConfig
from common.schemas import Call, Step
from common.seatbelt import SELF_CHECK_PROMPT, SELF_CHECK_RETRY, SELF_CHECK_SCHEMA, Seatbelt, error_output

if TYPE_CHECKING:
    from gym_env.env import Env

THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
NOT_EXECUTED = error_output("not_executed", "Not executed: the step stopped after too many failed calls.")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # the raw JSON string the model wrote


@dataclass
class ModelReply:
    message: dict[str, Any]  # the assistant message, ready to append to the conversation
    text: str  # the answer without reasoning
    tool_calls: list[ToolCall]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


ModelFn = Callable[
    [list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any] | None], Awaitable[ModelReply]
]
ToolFn = Callable[[str, Any], Awaitable[tuple[int, Any]]]


@dataclass
class StepResult:
    step: Step
    messages: list[dict[str, Any]]  # the whole conversation, starting with the input messages
    answer: str  # text of the last answer without tool calls ("" if there was none)
    prompt_tokens: int
    completion_tokens: int
    latency_s: float  # wall time of the step

    @property
    def gave_up(self) -> bool:
        return self.step.gave_up


def strip_reasoning(content: str | None) -> str:
    """Drop <think> blocks (how Gym's model server returns reasoning) from a message's content."""
    return THINK.sub("", content or "").strip()


def parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    return [
        ToolCall(call["id"], call["function"]["name"], call["function"].get("arguments") or "")
        for call in message.get("tool_calls") or []
    ]


def assistant_message(raw: dict[str, Any]) -> dict[str, Any]:
    """The parts of a returned assistant message that go back into the conversation."""
    message: dict[str, Any] = {"role": "assistant", "content": raw.get("content") or ""}
    if raw.get("tool_calls"):
        message["tool_calls"] = raw["tool_calls"]
    return message


def tool_message(call_id: str, output: Any) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(output)}


def chat_tools(response_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Responses API tool definitions (gym_env.env.tool_schemas) in chat format."""
    return [
        {
            "type": "function",
            "function": {key: tool[key] for key in ("name", "description", "parameters") if key in tool},
        }
        for tool in response_tools
    ]


async def run_step(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: ModelFn,
    call_tool: ToolFn,
    *,
    max_steps: int,
    seatbelt: SeatbeltConfig | None,
    agent: str,
    model_name: str,
) -> StepResult:
    """Run one step. `tools` are chat-format definitions; `seatbelt=None` turns the seatbelt off."""
    start = time.perf_counter()
    conversation = list(messages)
    belt = Seatbelt(seatbelt, tools) if seatbelt else None
    calls: list[Call] = []
    tokens = [0, 0]
    answer = ""

    async def ask(
        history: list[dict[str, Any]], tools: list[dict[str, Any]] | None, schema: dict[str, Any] | None
    ) -> ModelReply:
        reply = await model(history, tools, schema)
        tokens[0] += reply.prompt_tokens
        tokens[1] += reply.completion_tokens
        return reply

    for turn in range(1, max_steps + 1):
        reply = await ask(conversation, tools, None)
        conversation.append(reply.message)

        if not reply.tool_calls:
            answer = reply.text
            if belt is None:
                break
            check = await ask([*conversation, {"role": "user", "content": SELF_CHECK_PROMPT}], None, SELF_CHECK_SCHEMA)
            verdict = belt.self_check(turn, check.text)
            if verdict.complete:
                break
            if belt.exhausted:
                belt.give_up(turn, "the self-check says the task is not complete and no retries are left")
                break
            conversation.append({"role": "user", "content": SELF_CHECK_RETRY.format(reason=verdict.reason)})
            continue

        for i, tool_call in enumerate(reply.tool_calls):
            call = await _run_call(turn, tool_call, call_tool, belt)
            calls.append(call)
            conversation.append(tool_message(tool_call.id, call.output))
            if belt and belt.exhausted:
                # Answer the rest of this turn's calls so the conversation stays well formed.
                conversation += [tool_message(skipped.id, NOT_EXECUTED) for skipped in reply.tool_calls[i + 1 :]]
                belt.give_up(turn, "retry limit: too many invalid calls, errors or failed self-checks")
                break
        if belt and belt.exhausted:
            break
    else:
        if belt:
            belt.give_up(max_steps, f"max steps: no accepted final answer after {max_steps} turns")

    step = Step(agent=agent, model=model_name, calls=calls, seatbelt_events=belt.events if belt else [])
    return StepResult(step, conversation, answer, tokens[0], tokens[1], time.perf_counter() - start)


async def _run_call(turn: int, tool_call: ToolCall, call_tool: ToolFn, belt: Seatbelt | None) -> Call:
    name = tool_call.name
    if belt is None:
        try:
            args = json.loads(tool_call.arguments) if tool_call.arguments.strip() else {}
        except json.JSONDecodeError:
            args = tool_call.arguments  # the environment rejects it with a readable error
        status, output = await call_tool(name, args)
        return Call(turn=turn, tool=name, args=args, status=status, output=output)

    args, error = belt.check_call(turn, name, tool_call.arguments)
    if error is not None:
        return Call(turn=turn, tool=name, args=args, status=0, output=error)
    status, output = await call_tool(name, args)
    for attempt, delay in enumerate(belt.transient_delays(), start=1):
        if status != 503:
            break
        belt.transient_retry(turn, name, attempt, delay)
        await asyncio.sleep(delay)
        status, output = await call_tool(name, args)
    if status != 200:
        belt.tool_error(turn, name, status, output)
    return Call(turn=turn, tool=name, args=args, status=status, output=output)


def llm_model(model: str, sampling: dict[str, Any]) -> ModelFn:
    """A ModelFn on common/llm.py, for code outside Gym (`sampling` from EvalConfig.sampling())."""

    async def call(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, json_schema: dict[str, Any] | None
    ) -> ModelReply:
        thinking = False if json_schema is not None else None  # see the module docstring
        response = await asyncio.to_thread(
            llm.chat, model, messages, tools=tools, json_schema=json_schema, thinking=thinking, **sampling
        )
        message = assistant_message(response.message)
        # vLLM returns reasoning as `reasoning`, but both chat templates read `reasoning_content` from history.
        if response.message.get("reasoning"):
            message["reasoning_content"] = response.message["reasoning"]
        return ModelReply(
            message=message,
            text=strip_reasoning(message.get("content")),
            tool_calls=parse_tool_calls(message),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_s=response.latency_s,
        )

    return call


def env_tools(env: "Env") -> ToolFn:
    """A ToolFn on a gym_env Env, without HTTP."""

    async def call(name: str, args: Any) -> tuple[int, Any]:
        result = env.call(name, args)
        return result.status, result.output

    return call
