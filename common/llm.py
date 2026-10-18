"""Every model call outside NeMo Gym, with token and latency logging and live, record and replay modes.

LLM_MODE (environment variable):
- live (default): call the vLLM server.
- record: call the server and save the response to runs/cache/.
- replay: return the saved response for an identical request, without a server. A request that
  wasn't recorded raises ReplayMiss. Never evaluate in replay mode.

A request is identified by everything sent to the server (model, messages, tools, schema, thinking,
sampling), so identical sampled requests share one cache entry: the last one recorded wins.

LLM_CACHE_TAG (optional environment variable): when set, it's part of that identity too, so recordings of different
scenarios that send identical requests don't overwrite each other (the demo runs the same request with and without
chaos, task 5.3). Unset, requests are identified as before, and earlier recordings still replay.
"""

import hashlib
import json
import os
import time
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
MODELS_CONFIG = REPO / "configs" / "models.yaml"
CACHE_DIR = REPO / "runs" / "cache"
MODES = ("live", "record", "replay")


class ReplayMiss(KeyError):
    """LLM_MODE=replay and this exact request was never recorded."""


class LLMResponse(BaseModel):
    message: dict[str, Any]  # the assistant message: role, content, reasoning, tool_calls
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    replayed: bool = False  # served from runs/cache/; latency_s is the recorded one


@cache
def _models() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(MODELS_CONFIG.read_text())["models"]


def port_for(model: str) -> int:
    """Models in configs/models.yaml use their own port; any other name is a LoRA adapter on the
    student server."""
    models = _models()
    return models[model]["port"] if model in models else models["student"]["port"]


@cache
def _client(port: int) -> OpenAI:
    return OpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")


def _mode() -> str:
    mode = os.environ.get("LLM_MODE", "live")
    if mode not in MODES:
        raise ValueError(f"LLM_MODE={mode!r}; expected one of {MODES}")
    return mode


def _request(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    json_schema: dict[str, Any] | None,
    thinking: bool | None,
    sampling: dict[str, Any],
) -> dict[str, Any]:
    request: dict[str, Any] = {"model": model, "messages": messages, **sampling}
    if tools:
        request["tools"] = tools
    if json_schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "output", "schema": json_schema, "strict": True},
        }
    if thinking is not None:
        request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": thinking}}
    return request


def _cache_path(request: dict[str, Any]) -> Path:
    tag = os.environ.get("LLM_CACHE_TAG")
    keyed = {"cache_tag": tag, **request} if tag else request
    key = hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"


def chat(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    json_schema: dict[str, Any] | None = None,
    thinking: bool | None = None,
    **sampling: Any,
) -> LLMResponse:
    """One chat completion from `model` ("lightning", "student" or an adapter name).

    tools: OpenAI-format tool definitions. json_schema: constrain the answer to this JSON schema
    (vLLM structured outputs). thinking: None keeps the chat template's default (on for both of
    our models). sampling: passed through, e.g. temperature, top_p, max_tokens, seed.
    """
    mode = _mode()
    request = _request(model, messages, tools, json_schema, thinking, sampling)
    path = _cache_path(request)

    if mode == "replay":
        if not path.exists():
            raise ReplayMiss(f"no recorded response for this {model} request ({path.name})")
        response = LLMResponse.model_validate(json.loads(path.read_text())["response"])
        return response.model_copy(update={"replayed": True})

    start = time.perf_counter()
    completion = _client(port_for(model)).chat.completions.create(**request)
    latency_s = time.perf_counter() - start
    choice = completion.choices[0]
    response = LLMResponse(
        message=choice.message.model_dump(exclude_none=True),
        finish_reason=choice.finish_reason,
        prompt_tokens=completion.usage.prompt_tokens,
        completion_tokens=completion.usage.completion_tokens,
        latency_s=latency_s,
    )

    if mode == "record":
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"request": request, "response": response.model_dump()}, indent=1))
        tmp.replace(path)
    return response
