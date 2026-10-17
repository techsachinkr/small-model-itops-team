"""The Gym resources server over HTTP: sessions, tool routes, chaos and /verify."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("nemo_gym")  # Linux only (CLAUDE.md "Stack")

from fastapi.testclient import TestClient  # noqa: E402
from httpx import Cookies  # noqa: E402

from gym_env.chaos import ChaosConfig  # noqa: E402
from gym_env.server import ItopsResourcesServer, ItopsResourcesServerConfig  # noqa: E402
from gym_env.tasks import case_to_row, read_cases  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402

CASES = {case.id: case for case in read_cases(Path("resources_servers/itops/data/example_cases.jsonl"))}


class Session:
    """One Gym rollout: the cookie returned by /seed_session identifies the session."""

    def __init__(self, client: TestClient, row: dict[str, Any]) -> None:
        self.client, self.row = client, row
        response = client.post("/seed_session", json=row)
        assert response.status_code == 200
        self.cookies = response.cookies

    def call(self, tool: str, args: Any) -> Any:
        return self.client.post(f"/{tool}", json=args, cookies=self.cookies)

    def verify(self, answer: str = "Done.") -> dict[str, Any]:
        response = {
            "id": "resp-1", "created_at": 0, "model": "test", "object": "response",
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "output": [{"id": "msg-1", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": answer, "annotations": []}]}],
        }
        result = self.client.post("/verify", json=self.row | {"response": response}, cookies=self.cookies)
        assert result.status_code == 200, result.text
        return result.json()


def make_client(chaos: ChaosConfig | None = None) -> TestClient:
    config = ItopsResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="itops",
                                        chaos=chaos or ChaosConfig())
    server = ItopsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    client = TestClient(server.setup_webserver())

    class StatelessCookies(Cookies):  # as in Gym's own server tests: cookies only when passed
        def extract_cookies(self, response: Any) -> None:
            pass

    client._cookies = StatelessCookies(client._cookies)
    return client


def test_gold_run_scores_1_and_wrong_run_scores_0() -> None:
    client = make_client()
    case = CASES["ex-multi-001"]
    good, bad = Session(client, case_to_row(case)), Session(client, case_to_row(case))
    for call in case.gold_calls:
        assert good.call(call.tool, call.args).status_code == 200
    bad.call("create_ticket", case.gold_calls[0].args)

    result = good.verify()
    assert result["reward"] == 1.0 and result["state_diff"] == []
    result = bad.verify()
    assert result["reward"] == 0.0
    assert any("deployments.prod/payments" in line for line in result["state_diff"])


def test_sessions_are_isolated() -> None:
    client = make_client()
    row = case_to_row(CASES["ex-hd-001"])
    first, second = Session(client, row), Session(client, row)
    args = {"employee_id": "E1001", "category": "access", "summary": "VPN", "priority": "P3"}
    assert first.call("create_ticket", args).json()["ticket"]["id"] == "HD-1042"
    assert second.call("create_ticket", args).json()["ticket"]["id"] == "HD-1042"
    assert first.call("create_ticket", args).json()["ticket"]["id"] == "HD-1043"


def test_tool_errors_are_readable_json() -> None:
    session = Session(make_client(), case_to_row(CASES["ex-multi-001"]))
    response = session.call("rollback", {"namespace": "prod", "deployment": "payments", "to_revision": 8})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "change_ticket_required"
    response = session.call("drop_database", {})
    assert response.status_code == 404
    assert "Available tools" in response.json()["error"]["message"]
    response = session.client.post("/get_ticket", content=b"{not json", cookies=session.cookies)
    assert response.json()["error"]["code"] == "invalid_arguments"


def test_tool_call_before_seed_session_is_an_error() -> None:
    response = make_client().post("/list_devices", json={"employee_id": "E1001"})
    assert response.json()["error"]["code"] == "session_not_initialized"


def test_chaos_from_server_config_returns_503() -> None:
    session = Session(make_client(ChaosConfig(enabled=True, rate=1.0)), case_to_row(CASES["ex-hd-001"]))
    response = session.call("list_devices", {"employee_id": "E1004"})
    assert response.status_code == 503
    assert response.json()["error"]["retryable"] is True


def test_rows_carry_the_acting_user_and_only_the_case_apis_tools() -> None:
    row = case_to_row(CASES["ex-hd-001"])
    system, user = row["responses_create_params"]["input"]
    assert "Current user: E1004" in system["content"]
    assert "Monday 2026-10-19 09:00 UTC" in system["content"]
    assert user["content"] == CASES["ex-hd-001"].request
    names = {tool["name"] for tool in row["responses_create_params"]["tools"]}
    assert names == {"list_devices", "create_ticket", "list_tickets", "get_ticket", "update_ticket",
                     "reset_password"}
    for tool in row["responses_create_params"]["tools"]:
        assert "title" not in str(tool["parameters"])
        assert tool["parameters"]["additionalProperties"] is False
    (create,) = [t for t in row["responses_create_params"]["tools"] if t["name"] == "create_ticket"]
    category = create["parameters"]["properties"]["category"]
    for name in category["enum"]:  # every category is defined for the model
        assert f"{name}:" in category["description"]
