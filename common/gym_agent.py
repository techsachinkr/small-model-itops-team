"""NeMo Gym agent that runs common/tool_loop.py, with the seatbelt switched on or off by its config.

Gym only finds agents under responses_api_agents/<name>/, so responses_api_agents/itops_agent/app.py
starts this one. For each rollout (/run): seed the resources server's session from the row, run one
step of the tool loop, turn the conversation back into a Responses API response, and have the
resources server verify it. Model calls go to Gym's model server as chat completions, and tool calls
to the resources server. Sampling, max steps and seatbelt limits come from the eval config
(configs/eval.yaml), so every setup runs under the same conditions (hard rule 7); the sampling fields
of a row are ignored.

The verify response gets a `trace` field (the step, tokens and latency). A step the seatbelt gave up
on scores 0, because a failed step fails the case (CLAUDE.md "Seatbelt"); `verifier_reward` keeps the
state-match result.
"""

import json
from functools import cache
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict

from common.eval_config import REPO, EvalConfig, load_eval_config
from common.seatbelt import error_output
from common.tool_loop import ModelReply, StepResult, assistant_message, parse_tool_calls, run_step, strip_reasoning
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.server_utils import get_response_json, raise_for_status


class ItopsAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    model_name: str  # student, lightning or an adapter: recorded in the trace and used for prices
    seatbelt: bool  # off only for base_raw
    eval_config: str = "configs/eval.yaml"  # relative to the repo root
    agent_label: str = "single"  # Step.agent in the trace


class ItopsAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ItopsAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ItopsAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@cache
def eval_config(path: str) -> EvalConfig:
    return load_eval_config(REPO / path)


class ItopsAgent(SimpleResponsesAPIAgent):
    config: ItopsAgentConfig

    async def _episode(
        self, params: NeMoGymResponseCreateParamsNonStreaming, cookies: Any
    ) -> tuple[NeMoGymResponse, StepResult, Any]:
        settings = eval_config(self.config.eval_config)
        converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=True)
        chat = converter.responses_to_chat_completion_create_params(params)
        messages = [dict(message) for message in chat.messages]
        tools = [dict(tool) for tool in chat.tools or []]
        tool_names = [tool["function"]["name"] for tool in tools]
        session = {"resources": cookies, "model": None}

        async def model(
            history: list[dict[str, Any]], tools: list[dict[str, Any]] | None, schema: dict[str, Any] | None
        ) -> ModelReply:
            body: dict[str, Any] = {"messages": history, **settings.sampling()}
            if tools:
                body["tools"] = tools
            if schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "output", "schema": schema, "strict": True},
                }
            # Thinking as the eval config says, but always off for structured answers (common/tool_loop.py).
            # Gym's model server takes per-request chat template arguments from metadata.
            thinking = settings.thinking and schema is None
            body["metadata"] = {"chat_template_kwargs": json.dumps({"enable_thinking": thinking})}
            start = perf_counter()
            response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/chat/completions",
                json=body,
                cookies=session["model"],
            )
            await raise_for_status(response)  # model calls are expected to work, as in Gym's simple agent
            completion = await get_response_json(response)
            session["model"] = response.cookies
            # Gym's model server returns reasoning as <think>...</think> in the content and turns it back
            # into reasoning_content when the message is sent again, so the message goes back as it came.
            message = assistant_message(completion["choices"][0]["message"])
            usage = completion.get("usage") or {}
            return ModelReply(
                message=message,
                text=strip_reasoning(message["content"]),
                tool_calls=parse_tool_calls(message),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_s=perf_counter() - start,
            )

        async def call_tool(name: str, args: Any) -> tuple[int, Any]:
            if name not in tool_names:  # never post to the server's own routes, like /verify
                return 404, error_output("unknown_tool", f"Unknown tool {name!r}. Available tools: {', '.join(tool_names)}.")
            response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=f"/{name}",
                json=args,
                cookies=session["resources"],
            )
            session["resources"] = response.cookies
            raw = await response.read()
            try:
                output = json.loads(raw)
            except json.JSONDecodeError:
                output = raw.decode(errors="replace")
            return response.status, output

        result = await run_step(
            messages,
            tools,
            model,
            call_tool,
            max_steps=settings.max_steps,
            seatbelt=settings.seatbelt if self.config.seatbelt else None,
            agent=self.config.agent_label,
            model_name=self.config.model_name,
        )
        output = converter.chat_completions_messages_to_responses_items(result.messages[len(messages) :])
        response = NeMoGymResponse(
            id=f"resp_{uuid4().hex}",
            created_at=int(time()),
            model=self.config.model_name,
            object="response",
            output=output,
            parallel_tool_calls=params.parallel_tool_calls,
            tool_choice=params.tool_choice if params.tool_choice is not None else "auto",
            tools=params.tools,
            temperature=settings.temperature,
            top_p=settings.top_p,
            usage=NeMoGymResponseUsage(
                input_tokens=result.prompt_tokens,
                input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
                output_tokens=result.completion_tokens,
                output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )
        return response, result, session["resources"]

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """Run the loop on a request, without seeding or verifying; the caller's cookies pick the session."""
        model_response, _, cookies = await self._episode(body, request.cookies)
        for key, value in cookies.items():
            response.set_cookie(key, value)
        return model_response

    async def run(self, request: Request, body: ItopsAgentRunRequest) -> ItopsAgentVerifyResponse:
        resources = self.config.resources_server.name
        seeded = await self.server_client.post(
            server_name=resources, url_path="/seed_session", json=body.model_dump(), cookies=request.cookies
        )
        await raise_for_status(seeded)

        response, result, cookies = await self._episode(body.responses_create_params, seeded.cookies)

        verify_request = ItopsAgentVerifyRequest.model_validate(body.model_dump() | {"response": response})
        verified = await self.server_client.post(
            server_name=resources, url_path="/verify", json=verify_request.model_dump(), cookies=cookies
        )
        await raise_for_status(verified)
        verdict = await get_response_json(verified)
        verdict["verifier_reward"] = verdict["reward"]
        if result.gave_up:
            verdict["reward"] = 0.0
        verdict["trace"] = {
            "steps": [result.step.model_dump(mode="json")],
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_s": result.latency_s,
        }
        return ItopsAgentVerifyResponse.model_validate(verdict)
