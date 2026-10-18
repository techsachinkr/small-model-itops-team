"""NeMo Gym agents: ItopsAgent runs common/tool_loop.py with one model, the seatbelt switched on or off by its
config; ItopsTeamAgent runs the team (agents/orchestrator.py), with the seatbelt, and with escalation to Lightning
only when its config sets `escalate` (team_escalate, hard rule 7).

Gym only finds agents under responses_api_agents/<name>/, so responses_api_agents/itops_agent/app.py and
responses_api_agents/itops_team/app.py start these. For each rollout (/run): seed the resources server's
session from the row, run one step of the tool loop (or the team's plan and steps), turn the conversation back
into a Responses API response, and have the resources server verify it. Model calls go to Gym's model servers
as chat completions (the team's planner and each specialist have their own), and tool calls to the resources
server. Sampling, max steps and seatbelt limits come from the eval config (configs/eval.yaml), so every setup
runs under the same conditions (hard rule 7); the sampling fields of a row are ignored.

The verify response gets a `trace` field (the steps, the team's plan, tokens and latency). A step the seatbelt
gave up on scores 0, because a failed step fails the case (CLAUDE.md "Seatbelt"), and so does a team attempt
without a valid plan; `verifier_reward` keeps the state-match result.
"""

import json
from functools import cache
from time import perf_counter, time
from typing import Any
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict

from agents.orchestrator import Escalation, Member, TeamResult, run_team
from agents.specialist import Specialist, active_specialists
from common.eval_config import REPO, EvalConfig, load_eval_config
from common.seatbelt import error_output
from common.tool_loop import (
    ModelFn,
    ModelReply,
    StepResult,
    ToolFn,
    assistant_message,
    parse_tool_calls,
    run_step,
    strip_reasoning,
)
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
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


class ItopsAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    model_name: str  # student, lightning or an adapter: recorded in the trace and used for prices
    seatbelt: bool  # off only for base_raw
    eval_config: str = "configs/eval.yaml"  # relative to the repo root
    agent_label: str = "single"  # Step.agent in the trace


class ItopsTeamAgentConfig(BaseResponsesAPIAgentConfig):
    """One model server per specialist (<name>_server, <name>_model_name), as top-level fields so that Gym
    checks at startup that each one exists."""

    resources_server: ResourcesServerRef
    planner_server: ModelServerRef
    planner_model_name: str  # recorded in the plan and used for prices
    helpdesk_server: ModelServerRef
    helpdesk_model_name: str  # the student or an adapter, like helpdesk_r1
    ops_server: ModelServerRef
    ops_model_name: str
    # Escalation (CLAUDE.md "Escalation"): Lightning finishes a step the seatbelt gave up on, only when `escalate` is
    # set (team_escalate). The server is configured for every team setup (gym_configs/eval_team.yaml).
    escalate: bool = False
    escalation_server: ModelServerRef
    escalation_model_name: str  # recorded in the escalated step and used for its part's prices
    eval_config: str = "configs/eval.yaml"  # relative to the repo root
    registry: str = "agents/registry.json"  # which specialists are active


class ItopsAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ItopsAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ItopsAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@cache
def eval_config(path: str) -> EvalConfig:
    return load_eval_config(REPO / path)


def chat_request(
    params: NeMoGymResponseCreateParamsNonStreaming,
) -> tuple[ResponsesConverter, list[dict[str, Any]], list[dict[str, Any]]]:
    """The converter, and the row's input messages and tools in chat format."""
    converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=True)
    chat = converter.responses_to_chat_completion_create_params(params)
    return converter, [dict(message) for message in chat.messages], [dict(tool) for tool in chat.tools or []]


def gym_model(server_client: ServerClient, server_name: str, settings: EvalConfig) -> ModelFn:
    """A ModelFn on a Gym model server, keeping that server's cookies across the rollout's calls."""
    cookies: dict[str, Any] = {"model": None}

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
        response = await server_client.post(
            server_name=server_name, url_path="/v1/chat/completions", json=body, cookies=cookies["model"]
        )
        await raise_for_status(response)  # model calls are expected to work, as in Gym's simple agent
        completion = await get_response_json(response)
        cookies["model"] = response.cookies
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

    return model


def gym_tools(server_client: ServerClient, server_name: str, tool_names: list[str], session: dict[str, Any]) -> ToolFn:
    """A ToolFn on the resources server. session["resources"] holds the rollout's cookies, kept current."""

    async def call_tool(name: str, args: Any) -> tuple[int, Any]:
        if name not in tool_names:  # never post to the server's own routes, like /verify
            return 404, error_output("unknown_tool", f"Unknown tool {name!r}. Available tools: {', '.join(tool_names)}.")
        response = await server_client.post(
            server_name=server_name, url_path=f"/{name}", json=args, cookies=session["resources"]
        )
        session["resources"] = response.cookies
        raw = await response.read()
        try:
            output = json.loads(raw)
        except json.JSONDecodeError:
            output = raw.decode(errors="replace")
        return response.status, output

    return call_tool


def tool_names(tools: list[dict[str, Any]]) -> list[str]:
    return [tool["function"]["name"] for tool in tools]


def make_response(
    params: NeMoGymResponseCreateParamsNonStreaming, model: str, output: list[Any], settings: EvalConfig,
    prompt_tokens: int, completion_tokens: int,
) -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model=model,
        object="response",
        output=output,
        parallel_tool_calls=params.parallel_tool_calls,
        tool_choice=params.tool_choice if params.tool_choice is not None else "auto",
        tools=params.tools,
        temperature=settings.temperature,
        top_p=settings.top_p,
        usage=NeMoGymResponseUsage(
            input_tokens=prompt_tokens,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=completion_tokens,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def seed_session(server_client: ServerClient, resources: str, body: ItopsAgentRunRequest, cookies: Any) -> Any:
    """Seed a fresh session from the row; returns its cookies."""
    seeded = await server_client.post(
        server_name=resources, url_path="/seed_session", json=body.model_dump(), cookies=cookies
    )
    await raise_for_status(seeded)
    return seeded.cookies


async def verify(
    server_client: ServerClient, resources: str, body: ItopsAgentRunRequest, response: NeMoGymResponse,
    cookies: Any, trace: dict[str, Any], failed: bool,
) -> ItopsAgentVerifyResponse:
    """Have the resources server score the session; a failed attempt scores 0 whatever the state."""
    verify_request = ItopsAgentVerifyRequest.model_validate(body.model_dump() | {"response": response})
    verified = await server_client.post(
        server_name=resources, url_path="/verify", json=verify_request.model_dump(), cookies=cookies
    )
    await raise_for_status(verified)
    verdict = await get_response_json(verified)
    verdict["verifier_reward"] = verdict["reward"]
    if failed:
        verdict["reward"] = 0.0
    verdict["trace"] = trace
    return ItopsAgentVerifyResponse.model_validate(verdict)


class ItopsAgent(SimpleResponsesAPIAgent):
    config: ItopsAgentConfig

    async def _episode(
        self, params: NeMoGymResponseCreateParamsNonStreaming, cookies: Any
    ) -> tuple[NeMoGymResponse, StepResult, Any]:
        settings = eval_config(self.config.eval_config)
        converter, messages, tools = chat_request(params)
        session = {"resources": cookies}
        result = await run_step(
            messages,
            tools,
            gym_model(self.server_client, self.config.model_server.name, settings),
            gym_tools(self.server_client, self.config.resources_server.name, tool_names(tools), session),
            max_steps=settings.max_steps,
            seatbelt=settings.seatbelt if self.config.seatbelt else None,
            agent=self.config.agent_label,
            model_name=self.config.model_name,
        )
        output = converter.chat_completions_messages_to_responses_items(result.messages[len(messages) :])
        response = make_response(
            params, self.config.model_name, output, settings, result.prompt_tokens, result.completion_tokens
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
        cookies = await seed_session(self.server_client, resources, body, request.cookies)
        response, result, cookies = await self._episode(body.responses_create_params, cookies)
        trace = {
            "steps": [result.step.model_dump(mode="json")],
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_s": result.latency_s,
        }
        return await verify(self.server_client, resources, body, response, cookies, trace, failed=result.gave_up)


def text_of(content: Any) -> str:
    """A chat message's content as text (a string, or a list of text parts)."""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


class ItopsTeamAgent(SimpleResponsesAPIAgent):
    config: ItopsTeamAgentConfig

    def _member(self, specialist: Specialist, settings: EvalConfig) -> Member:
        server: ModelServerRef | None = getattr(self.config, f"{specialist.name}_server", None)
        if server is None:
            raise ValueError(f"The team agent has no model server for specialist {specialist.name!r}: add "
                             f"{specialist.name}_server and {specialist.name}_model_name to its config.")
        model_name = getattr(self.config, f"{specialist.name}_model_name")
        return Member(specialist, gym_model(self.server_client, server.name, settings), model_name)

    async def _episode(
        self, params: NeMoGymResponseCreateParamsNonStreaming, cookies: Any
    ) -> tuple[NeMoGymResponse, TeamResult, Any]:
        settings = eval_config(self.config.eval_config)
        converter, messages, tools = chat_request(params)
        system = "\n".join(text_of(m["content"]) for m in messages if m["role"] == "system")
        request = [text_of(m["content"]) for m in messages if m["role"] == "user"][-1]
        session = {"resources": cookies}
        escalation = None
        if self.config.escalate:
            escalation = Escalation(gym_model(self.server_client, self.config.escalation_server.name, settings),
                                    self.config.escalation_model_name)
        result = await run_team(
            request,
            system,
            tools,
            gym_model(self.server_client, self.config.planner_server.name, settings),
            self.config.planner_model_name,
            [self._member(specialist, settings) for specialist in active_specialists(REPO / self.config.registry)],
            gym_tools(self.server_client, self.config.resources_server.name, tool_names(tools), session),
            max_steps=settings.max_steps,
            seatbelt=settings.seatbelt,
            escalation=escalation,
        )
        # Each step's turns after its system prompt and instruction (for an escalated step, the specialist's, the
        # hand-off and Lightning's, each once), then the team's answer as the last message, which is what the
        # verifier reads.
        output = []
        for step in result.steps:
            output += converter.chat_completions_messages_to_responses_items(step.messages[2:])
        output += converter.chat_completions_messages_to_responses_items([{"role": "assistant", "content": result.answer}])
        response = make_response(
            params, self.config.planner_model_name, output, settings, result.prompt_tokens, result.completion_tokens
        )
        return response, result, session["resources"]

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        """Run the team on a request, without seeding or verifying; the caller's cookies pick the session."""
        model_response, _, cookies = await self._episode(body, request.cookies)
        for key, value in cookies.items():
            response.set_cookie(key, value)
        return model_response

    async def run(self, request: Request, body: ItopsAgentRunRequest) -> ItopsAgentVerifyResponse:
        resources = self.config.resources_server.name
        cookies = await seed_session(self.server_client, resources, body, request.cookies)
        response, result, cookies = await self._episode(body.responses_create_params, cookies)
        trace = {
            "plan": result.plan.model_dump(mode="json"),
            "steps": [step.step.model_dump(mode="json") for step in result.steps],
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_s": result.latency_s,
        }
        return await verify(self.server_client, resources, body, response, cookies, trace, failed=result.failed)
