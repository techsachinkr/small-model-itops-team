"""NeMo Gym resources server for the helpdesk and ops tools.

Follows Gym's Stateful Environment pattern: /seed_session builds a fresh Env for the session from the
row's case, every tool is POST /<tool_name> on that session's Env, and /verify scores the session's
final state with gym_env/verify.py, then drops the session. Gym only finds servers under
resources_servers/<name>/, so resources_servers/itops/app.py starts this one.
"""

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, Field

from common.schemas import Case
from gym_env.chaos import ChaosConfig
from gym_env.env import Env
from gym_env.tools import ToolError
from gym_env.verify import check
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import SESSION_ID_KEY


class ItopsResourcesServerConfig(BaseResourcesServerConfig):
    chaos: ChaosConfig = Field(default_factory=ChaosConfig)


class ItopsSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    case: Case


class ItopsVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    case: Case


class ItopsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    state_match: bool
    answer_ok: bool
    state_diff: list[str]
    missing_from_answer: list[str]
    forbidden_in_answer: list[str]


def final_answer(response: NeMoGymResponse) -> str:
    """Text of the last assistant message in the episode."""
    answer = ""
    for item in response.output:
        if item.type == "message" and item.role == "assistant":
            answer = "\n".join(part.text for part in item.content if part.type == "output_text")
    return answer


class ItopsResourcesServer(SimpleResourcesServer):
    config: ItopsResourcesServerConfig
    session_id_to_env: dict[str, Any] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Registered after /seed_session, /verify and /aggregate_metrics, so it doesn't shadow them.
        app.post("/{tool_name}")(self.call_tool)
        return app

    async def seed_session(self, request: Request, body: ItopsSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        # Assign, don't setdefault: a re-seed must start from a fresh fixture.
        self.session_id_to_env[session_id] = Env(
            body.case.fixture, apis=body.case.apis, chaos=self.config.chaos, chaos_key=body.case.id
        )
        return BaseSeedSessionResponse()

    async def call_tool(self, tool_name: str, request: Request) -> JSONResponse:
        env: Env | None = self.session_id_to_env.get(request.session[SESSION_ID_KEY])
        if env is None:
            error = ToolError("session_not_initialized", "Session not initialized: call /seed_session first.")
            return JSONResponse(error.to_dict(), status_code=error.status)
        raw = await request.body()
        try:
            args = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            args = raw.decode(errors="replace")  # rejected by validation with a readable message
        result = env.call(tool_name, args)
        return JSONResponse(result.output, status_code=result.status)

    async def verify(self, request: Request, body: ItopsVerifyRequest) -> ItopsVerifyResponse:
        env: Env | None = self.session_id_to_env.pop(request.session[SESSION_ID_KEY], None)
        if env is None:
            return ItopsVerifyResponse(
                **body.model_dump(), reward=0.0, failure_reason="session was never seeded",
                state_match=False, answer_ok=False, state_diff=[], missing_from_answer=[], forbidden_in_answer=[],
            )
        verdict = check(body.case, env.state, final_answer(body.response))
        return ItopsVerifyResponse(
            **body.model_dump(), reward=float(verdict.passed), **verdict.model_dump(exclude={"passed"})
        )
