"""The NeMo Gym agents in process: /run of ItopsAgent and ItopsTeamAgent against the real resources server (through
its FastAPI app) and scripted model servers, without starting Gym. Checks what the eval runs depend on: one shared
session, the tools each model gets, the planner's structured request, the response the verifier reads, the trace,
and failed attempts scoring 0."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("nemo_gym")  # Linux only (CLAUDE.md "Stack")

from fastapi.testclient import TestClient  # noqa: E402
from httpx import Cookies  # noqa: E402

from common.gym_agent import (  # noqa: E402
    ItopsAgent,
    ItopsAgentConfig,
    ItopsAgentRunRequest,
    ItopsTeamAgent,
    ItopsTeamAgentConfig,
)
from common.schemas import Case  # noqa: E402
from gym_env.env import tools_for  # noqa: E402
from gym_env.server import ItopsResourcesServer, ItopsResourcesServerConfig  # noqa: E402
from gym_env.tasks import case_to_row, read_cases  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402

CASES = {case.id: case for path in sorted(Path("data/practice").glob("*.jsonl")) for case in read_cases(path)}
SELF_CHECK_OK = json.dumps({"reason": "done", "tool_call_needed": False})


def resources_client() -> TestClient:
    config = ItopsResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="itops")
    client = TestClient(ItopsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient)).setup_webserver())

    class StatelessCookies(Cookies):  # as in tests/test_gym_server.py: cookies only when passed
        def extract_cookies(self, response: Any) -> None:
            pass

    client._cookies = StatelessCookies(client._cookies)
    return client


class FakeResponse:
    """The parts of aiohttp's ClientResponse the agent uses."""

    def __init__(self, status: int, body: bytes, cookies: Any) -> None:
        self.status, self.ok, self.cookies, self._body = status, status < 400, cookies, body

    async def read(self) -> bytes:
        return self._body


def message(text: str = "", **calls: dict[str, Any]) -> dict[str, Any]:
    """A chat completion message: a text answer, or tool calls (tool name -> arguments)."""
    reply: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        reply["tool_calls"] = [{"id": f"call-{i}", "type": "function",
                                "function": {"name": name, "arguments": json.dumps(args)}}
                               for i, (name, args) in enumerate(calls.items())]
    return reply


def plan(*steps: tuple[str, str, list[int]]) -> dict[str, Any]:
    return message(json.dumps({"analysis": "test", "uncovered": [], "reply": "",
                               "steps": [{"agent": a, "instruction": i, "depends_on": d} for a, i, d in steps]}))


def run(agent_class: type, config: Any, case: Case, **scripts: list[dict[str, Any]]
        ) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Run an agent on a case; `scripts` are each model server's replies. Returns the verify response and every
    request each model server got."""
    resources = resources_client()
    requests: dict[str, list[dict[str, Any]]] = {name: [] for name in scripts}

    async def post(server_name: str, url_path: str, **kwargs: Any) -> FakeResponse:
        if server_name == "itops":
            response = resources.post(url_path, json=kwargs.get("json"), cookies=kwargs.get("cookies"))
            return FakeResponse(response.status_code, response.content, response.cookies)
        requests[server_name].append(kwargs["json"])
        completion = {"choices": [{"message": scripts[server_name].pop(0)}],
                      "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        return FakeResponse(200, json.dumps(completion).encode(), {})

    server_client = MagicMock(spec=ServerClient)
    server_client.post = post
    agent = agent_class(config=config, server_client=server_client)
    body = ItopsAgentRunRequest.model_validate(case_to_row(case))
    verdict = asyncio.run(agent.run(SimpleNamespace(cookies={}), body))
    return verdict.model_dump(), requests


SERVER = {"host": "0.0.0.0", "port": 8081, "entrypoint": ""}
RESOURCES = {"type": "resources_servers", "name": "itops"}


def run_team(case_id: str, **scripts: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    config = ItopsTeamAgentConfig(
        **SERVER, name="itops_team", resources_server=RESOURCES,
        planner_server={"type": "responses_api_models", "name": "planner_model"}, planner_model_name="lightning",
        helpdesk_server={"type": "responses_api_models", "name": "helpdesk_model"}, helpdesk_model_name="helpdesk_r1",
        ops_server={"type": "responses_api_models", "name": "ops_model"}, ops_model_name="ops_r1",
    )
    return run(ItopsTeamAgent, config, CASES[case_id], **scripts)


def test_the_single_agent_runs_one_step_and_scores_it() -> None:
    config = ItopsAgentConfig(**SERVER, name="itops_agent", resources_server=RESOURCES, model_name="ops_r1",
                              model_server={"type": "responses_api_models", "name": "policy_model"}, seatbelt=True)
    scale = {"namespace": "prod", "deployment": "payments", "replicas": 5, "change_ticket": "HD-1041"}
    verdict, requests = run(ItopsAgent, config, CASES["pr-ops-02"], policy_model=[
        message(list_deployments={"namespace": "prod"}), message(scale=scale), message("Payments now has 5 replicas."),
        message(SELF_CHECK_OK)])
    assert verdict["reward"] == 1.0 and verdict["state_diff"] == []
    assert {tool["function"]["name"] for tool in requests["policy_model"][0]["tools"]} == {t.name for t in tools_for(["ops"])}
    assert json.loads(requests["policy_model"][-1]["metadata"]["chat_template_kwargs"]) == {"enable_thinking": False}
    (step,) = verdict["trace"]["steps"]
    assert (step["agent"], step["model"], step["prompt_tokens"]) == ("single", "ops_r1", 400)
    assert [call["tool"] for call in step["calls"]] == ["list_deployments", "scale"]
    assert verdict["trace"]["prompt_tokens"] == 400 and "plan" not in verdict["trace"]


def test_the_demo_request_through_the_gym_agent() -> None:
    ticket = {"employee_id": "E1012", "category": "incident", "summary": "Payments crashing in prod", "priority": "P1"}
    rollback = {"namespace": "prod", "deployment": "payments", "to_revision": 8, "change_ticket": "HD-1042"}
    verdict, requests = run_team(
        "pr-multi-01",
        planner_model=[plan(("helpdesk", "Open a P1 incident ticket for payments crashing in prod.", []),
                            ("ops", "Roll payments in prod back to its last healthy revision with step 1's ticket.", [1]))],
        helpdesk_model=[message(create_ticket=ticket), message("Opened HD-1042."), message(SELF_CHECK_OK)],
        ops_model=[message(rollout_history={"namespace": "prod", "deployment": "payments"}), message(rollback=rollback),
                   message("Rolled payments back to revision 8."), message(SELF_CHECK_OK)],
    )
    assert verdict["reward"] == 1.0 and verdict["verifier_reward"] == 1.0 and verdict["state_diff"] == []

    planner_request = requests["planner_model"][0]
    assert planner_request["response_format"]["json_schema"]["schema"]["properties"]["steps"]["maxItems"] == 6
    assert json.loads(planner_request["metadata"]["chat_template_kwargs"]) == {"enable_thinking": False}
    assert planner_request["temperature"] == 0.6 and "tools" not in planner_request
    for name, api in (("helpdesk_model", "helpdesk"), ("ops_model", "ops")):
        sent = {tool["function"]["name"] for tool in requests[name][0]["tools"]}
        assert sent == {tool.name for tool in tools_for([api])}
    assert "HD-1042" in requests["ops_model"][0]["messages"][1]["content"]  # step 1's result reached ops

    trace = verdict["trace"]
    assert [step["agent"] for step in trace["plan"]["steps"]] == ["helpdesk", "ops"]
    assert [(step["agent"], step["model"], step["prompt_tokens"]) for step in trace["steps"]] == [
        ("helpdesk", "helpdesk_r1", 300), ("ops", "ops_r1", 400)]
    assert trace["prompt_tokens"] == 800 and trace["plan"]["prompt_tokens"] == 100
    last = verdict["response"]["output"][-1]
    assert last["role"] == "assistant" and last["content"][0]["text"] == "Opened HD-1042.\n\nRolled payments back to revision 8."


@pytest.mark.parametrize("planner, helpdesk", [
    ([plan(("helpdesk", "Look at the current user's tickets.", []))],
     [message(drop_tickets={}, close_all={}, purge={})]),  # three invalid calls: the step gives up
    ([message("not a plan")], []),
], ids=["failed step", "invalid plan"])
def test_a_failed_team_attempt_scores_0_whatever_the_state(planner: list[dict[str, Any]], helpdesk: list[dict[str, Any]]) -> None:
    verdict, _ = run_team("pr-multi-08", planner_model=planner, helpdesk_model=helpdesk, ops_model=[])
    assert verdict["verifier_reward"] == 1.0  # nothing changed, which is right for this case
    assert verdict["reward"] == 0.0
