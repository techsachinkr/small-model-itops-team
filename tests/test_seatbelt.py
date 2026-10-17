"""The seatbelt inside the tool loop: 503 recovery, invalid calls sent back, the retry limit and the
self-check, plus the loop with the seatbelt off. A scripted fake model plays against a real Env."""

import asyncio
import json
from typing import Any

from common.eval_config import SeatbeltConfig
from common.schemas import SeatbeltEvent
from common.seatbelt import SELF_CHECK_PROMPT
from common.tool_loop import ModelReply, StepResult, ToolCall, chat_tools, env_tools, run_step
from gym_env.chaos import unavailable
from gym_env.env import Env, tool_schemas

CONFIG = SeatbeltConfig(max_retries_per_step=2, transient_retries=3, transient_backoff_s=0)
TOOLS = chat_tools(tool_schemas(["helpdesk"]))
MESSAGES = [{"role": "system", "content": "Current user: E1004"}, {"role": "user", "content": "Open a ticket."}]
TICKET = {"employee_id": "E1004", "category": "access", "summary": "VPN access", "priority": "P3"}


def turn(*calls: tuple[str, Any], text: str = "") -> ModelReply:
    """A model reply: tool calls (arguments as a dict, or a raw string), or a final answer."""
    tool_calls = [
        ToolCall(f"call-{i}", name, args if isinstance(args, str) else json.dumps(args))
        for i, (name, args) in enumerate(calls)
    ]
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in tool_calls
        ]
    return ModelReply(message, text, tool_calls, prompt_tokens=100, completion_tokens=10, latency_s=0.0)


def verdict(complete: bool, reason: str = "checked") -> ModelReply:
    return turn(text=json.dumps({"complete": complete, "reason": reason}))


class ScriptedModel:
    """Replies in order; records every request."""

    def __init__(self, *replies: ModelReply) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, messages: list[dict[str, Any]], tools: Any, json_schema: Any) -> ModelReply:
        self.requests.append({"messages": list(messages), "tools": tools, "json_schema": json_schema})
        return self.replies.pop(0)


def flaky(env: Env, failures: int):
    """Env tools whose first `failures` calls return a transient 503."""
    inner = env_tools(env)
    remaining = [failures]

    async def call(name: str, args: Any) -> tuple[int, Any]:
        if remaining[0] > 0:
            remaining[0] -= 1
            return 503, unavailable("helpdesk").to_dict()
        return await inner(name, args)

    return call


def run(model: ScriptedModel, tools: Any = None, seatbelt: SeatbeltConfig | None = CONFIG,
        env: Env | None = None, max_steps: int = 10) -> tuple[StepResult, Env]:
    env = env or Env("default", apis=["helpdesk"])
    result = asyncio.run(run_step(MESSAGES, TOOLS, model, tools or env_tools(env), max_steps=max_steps,
                                  seatbelt=seatbelt, agent="single", model_name="student"))
    return result, env


def kinds(events: list[SeatbeltEvent]) -> list[str]:
    return [event.kind for event in events]


def tool_results(result: StepResult) -> list[dict[str, Any]]:
    return [json.loads(m["content"]) for m in result.messages if m["role"] == "tool"]


def test_recovers_from_503s_without_the_model() -> None:
    env = Env("default", apis=["helpdesk"])
    model = ScriptedModel(turn(("create_ticket", TICKET)), turn(text="Opened HD-1042."), verdict(True))
    result, _ = run(model, tools=flaky(env, failures=2), env=env)

    assert not result.gave_up
    assert kinds(result.step.seatbelt_events) == ["transient_retry", "transient_retry", "self_check"]
    assert [c.status for c in result.step.calls] == [200]  # one call as the model sees it
    assert tool_results(result)[0]["ticket"]["id"] == "HD-1042"
    assert [t.id for t in env.state.tickets][-1] == "HD-1042"
    assert len(model.requests) == 3  # two turns and the self-check: the 503s never reached the model


def test_503_after_every_automatic_retry_goes_back_to_the_model_and_uses_a_retry() -> None:
    env = Env("default", apis=["helpdesk"])
    model = ScriptedModel(turn(("create_ticket", TICKET)), turn(("create_ticket", TICKET)),
                          turn(text="Opened HD-1042."), verdict(True))
    result, _ = run(model, tools=flaky(env, failures=4), env=env)

    assert kinds(result.step.seatbelt_events) == ["transient_retry"] * 3 + ["tool_error", "self_check"]
    assert result.step.seatbelt_events[3].retries_left == 1
    assert tool_results(result)[0]["error"]["code"] == "service_unavailable"
    assert [c.status for c in result.step.calls] == [503, 200]
    assert not result.gave_up


def test_invalid_arguments_are_sent_back_without_running_the_call() -> None:
    bad = TICKET | {"priority": "P5"}
    model = ScriptedModel(turn(("create_ticket", bad)), turn(("create_ticket", TICKET)),
                          turn(text="Opened HD-1042."), verdict(True))
    result, env = run(model)

    first = result.step.calls[0]
    assert first.status == 0 and first.args == bad
    assert first.output["error"]["code"] == "invalid_arguments"
    assert "priority" in first.output["error"]["message"] and "P5" in first.output["error"]["message"]
    assert [c.tool for c in env.calls] == ["create_ticket"]  # only the corrected call reached the env
    assert kinds(result.step.seatbelt_events) == ["invalid_call", "self_check"]
    assert not result.gave_up


def test_unknown_tools_and_bad_json_are_sent_back() -> None:
    model = ScriptedModel(turn(("delete_ticket", {"ticket_id": "HD-1035"}), ("create_ticket", "{priority: P3")),
                          turn(text="Done."), verdict(True))
    result, env = run(model)

    codes = [r["error"]["code"] for r in tool_results(result)]
    assert codes == ["unknown_tool", "invalid_json"]
    assert "Available tools" in tool_results(result)[0]["error"]["message"]
    assert result.step.calls[1].args == "{priority: P3"
    assert env.calls == []


def test_environment_errors_are_sent_back_and_use_a_retry() -> None:
    missing = {"ticket_id": "HD-9999"}  # valid by the schema, unknown to the env
    model = ScriptedModel(turn(("get_ticket", missing)), turn(text="No such ticket."), verdict(True))
    result, _ = run(model)

    assert result.step.calls[0].status == 400
    assert tool_results(result)[0]["error"]["code"] == "not_found"
    assert result.step.seatbelt_events[0].kind == "tool_error"
    assert result.step.seatbelt_events[0].retries_left == 1


def test_retry_limit_fails_the_step_on_the_third_failure() -> None:
    bad = TICKET | {"priority": "urgent"}
    model = ScriptedModel(turn(("create_ticket", bad)), turn(("create_ticket", bad)),
                          turn(("create_ticket", bad), ("list_tickets", {"employee_id": "E1004"})))
    result, env = run(model)

    assert result.gave_up
    assert kinds(result.step.seatbelt_events) == ["invalid_call"] * 3 + ["gave_up"]
    assert [e.retries_left for e in result.step.seatbelt_events[:3]] == [1, 0, 0]
    assert len(model.requests) == 3  # no self-check, no further turns
    assert tool_results(result)[-1]["error"]["code"] == "not_executed"  # the turn's other call never ran
    assert env.calls == []


def test_failed_self_check_continues_the_step() -> None:
    model = ScriptedModel(turn(text="Sure, I can do that."), verdict(False, "No ticket was opened."),
                          turn(("create_ticket", TICKET)), turn(text="Opened HD-1042."), verdict(True))
    result, env = run(model)

    check = model.requests[1]
    assert set(check["json_schema"]["required"]) == {"complete", "reason"} and check["tools"] is None
    assert check["messages"][-1] == {"role": "user", "content": SELF_CHECK_PROMPT}
    assert "No ticket was opened." in model.requests[2]["messages"][-1]["content"]
    assert SELF_CHECK_PROMPT not in json.dumps(result.messages)  # the check isn't part of the conversation
    assert result.answer == "Opened HD-1042."
    assert [e.complete for e in result.step.seatbelt_events] == [False, True]
    assert env.state.tickets[-1].id == "HD-1042" and not result.gave_up


def test_failed_self_check_with_no_retries_left_fails_the_step() -> None:
    model = ScriptedModel(*[reply for _ in range(3) for reply in (turn(text="Done."), verdict(False, "Nothing done."))])
    result, _ = run(model)

    assert result.gave_up
    assert kinds(result.step.seatbelt_events) == ["self_check"] * 3 + ["gave_up"]


def test_max_steps_fails_the_step() -> None:
    model = ScriptedModel(*[turn(("list_tickets", {"employee_id": "E1004"})) for _ in range(3)])
    result, _ = run(model, max_steps=3)

    assert result.gave_up and len(result.step.calls) == 3
    assert "max steps" in result.step.seatbelt_events[-1].detail


def test_tokens_include_the_self_check() -> None:
    model = ScriptedModel(turn(("create_ticket", TICKET)), turn(text="Opened HD-1042."), verdict(True))
    result, _ = run(model)
    assert (result.prompt_tokens, result.completion_tokens) == (300, 30)


def test_seatbelt_off_passes_everything_through() -> None:
    env = Env("default", apis=["helpdesk"])
    bad = TICKET | {"priority": "P5"}
    model = ScriptedModel(turn(("create_ticket", TICKET)), turn(("create_ticket", bad)),
                          turn(("create_ticket", "not json")), turn(text="Sorry, that failed."))
    result, _ = run(model, tools=flaky(env, failures=1), seatbelt=None, env=env)

    # The 503 isn't retried, and invalid arguments reach the env's own validation; every result goes
    # back to the model. No retries, no limit, no self-check, no events.
    assert [c.status for c in result.step.calls] == [503, 400, 400]
    assert "arguments must be a JSON object" in result.step.calls[2].output["error"]["message"]
    assert result.step.seatbelt_events == [] and not result.gave_up
    assert len(model.requests) == 4 and all(r["json_schema"] is None for r in model.requests)


def test_seatbelt_off_stops_at_max_steps_without_failing() -> None:
    model = ScriptedModel(*[turn(("list_tickets", {"employee_id": "E1004"})) for _ in range(2)])
    result, _ = run(model, seatbelt=None, max_steps=2)
    assert not result.gave_up and result.answer == "" and len(result.step.calls) == 2


def test_llm_model_renames_reasoning_and_turns_thinking_off_for_the_self_check(monkeypatch) -> None:
    from common import llm, tool_loop

    requests: list[dict[str, Any]] = []

    def fake_chat(model: str, messages: Any, **kwargs: Any) -> llm.LLMResponse:
        requests.append(kwargs)
        return llm.LLMResponse(message={"role": "assistant", "content": "Done.", "reasoning": "thought"},
                               finish_reason="stop", prompt_tokens=5, completion_tokens=2, latency_s=0.1)

    monkeypatch.setattr(llm, "chat", fake_chat)
    model = tool_loop.llm_model("student", {"temperature": 0.6})
    reply = asyncio.run(model(MESSAGES, TOOLS, None))
    asyncio.run(model(MESSAGES, None, {"type": "object"}))

    assert reply.message == {"role": "assistant", "content": "Done.", "reasoning_content": "thought"}
    assert reply.text == "Done." and reply.tool_calls == []
    assert requests[0]["thinking"] is None and requests[0]["temperature"] == 0.6
    assert requests[1]["thinking"] is False
