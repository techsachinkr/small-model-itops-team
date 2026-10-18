"""common/llm.py without a GPU: a stub client stands in for the vLLM servers."""

from typing import Any

import pytest
from openai.types.chat import ChatCompletion

from common import llm

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]
MESSAGES = [{"role": "user", "content": "Weather in Raleigh?"}]


class StubClient:
    """Records each create() call and answers with one parsed tool call."""

    def __init__(self, port: int, calls: list[tuple[int, dict[str, Any]]]) -> None:
        self.port = port
        self.calls = calls
        self.chat = self
        self.completions = self

    def create(self, **request: Any) -> ChatCompletion:
        self.calls.append((self.port, request))
        return ChatCompletion.model_validate({
            "id": "x", "object": "chat.completion", "created": 0, "model": request["model"],
            "choices": [{
                "index": 0, "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant", "content": None, "reasoning": "Need the weather.",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {
                        "name": "get_weather", "arguments": '{"city": "Raleigh"}'}}],
                },
            }],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
        })


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch, tmp_path) -> list[tuple[int, dict[str, Any]]]:
    calls: list[tuple[int, dict[str, Any]]] = []
    monkeypatch.setattr(llm, "_client", lambda port: StubClient(port, calls))
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    return calls


def test_live_returns_message_tokens_and_latency(calls, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    out = llm.chat("lightning", MESSAGES, tools=TOOLS, temperature=1.0)
    assert out.message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert out.message["reasoning"] == "Need the weather."
    assert (out.prompt_tokens, out.completion_tokens) == (120, 30)
    assert out.latency_s >= 0 and not out.replayed
    assert calls[0][1]["temperature"] == 1.0


def test_routing_by_model_name(calls, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    for model in ("lightning", "student", "ops_r1"):
        llm.chat(model, MESSAGES)
    assert [port for port, _ in calls] == [8000, 8001, 8001]
    assert calls[2][1]["model"] == "ops_r1"


def test_schema_and_thinking_reach_the_request(calls, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "live")
    schema = {"type": "object", "properties": {"complete": {"type": "boolean"}}}
    llm.chat("student", MESSAGES, json_schema=schema, thinking=False)
    request = calls[0][1]
    assert request["response_format"]["json_schema"]["schema"] == schema
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "tools" not in request


def test_record_then_replay_without_server(calls, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "record")
    recorded = llm.chat("student", MESSAGES, tools=TOOLS, seed=0)
    assert len(calls) == 1

    monkeypatch.setenv("LLM_MODE", "replay")
    replayed = llm.chat("student", MESSAGES, tools=TOOLS, seed=0)
    assert len(calls) == 1  # no server call
    assert replayed.replayed
    assert replayed.model_dump(exclude={"replayed"}) == recorded.model_dump(exclude={"replayed"})


def test_replay_miss_on_any_difference(calls, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "record")
    llm.chat("student", MESSAGES, tools=TOOLS, seed=0)
    monkeypatch.setenv("LLM_MODE", "replay")
    with pytest.raises(llm.ReplayMiss):
        llm.chat("student", MESSAGES, tools=TOOLS, seed=1)
    with pytest.raises(llm.ReplayMiss):
        llm.chat("lightning", MESSAGES, tools=TOOLS, seed=0)
    assert len(calls) == 1


def test_a_cache_tag_keeps_scenarios_apart_and_untagged_keys_unchanged(calls, monkeypatch, tmp_path):
    """The demo records the same request in different scenarios (with and without chaos); unset, the key is the
    request's own, so recordings made before the tag existed still replay."""
    request = llm._request("student", MESSAGES, TOOLS, None, None, {"seed": 0})
    untagged = llm._cache_path(request)
    assert untagged.name == f"{llm.hashlib.sha256(llm.json.dumps(request, sort_keys=True).encode()).hexdigest()}.json"
    monkeypatch.setenv("LLM_CACHE_TAG", "demo:with chaos")
    tagged = llm._cache_path(request)
    assert tagged != untagged
    monkeypatch.setenv("LLM_MODE", "record")
    llm.chat("student", MESSAGES, tools=TOOLS, seed=0)
    monkeypatch.setenv("LLM_MODE", "replay")
    assert llm.chat("student", MESSAGES, tools=TOOLS, seed=0).replayed
    monkeypatch.setenv("LLM_CACHE_TAG", "demo:another scenario")
    with pytest.raises(llm.ReplayMiss):
        llm.chat("student", MESSAGES, tools=TOOLS, seed=0)
    monkeypatch.delenv("LLM_CACHE_TAG")
    assert llm._cache_path(request) == untagged


def test_unknown_mode(calls, monkeypatch):
    monkeypatch.setenv("LLM_MODE", "offline")
    with pytest.raises(ValueError, match="LLM_MODE"):
        llm.chat("student", MESSAGES)
