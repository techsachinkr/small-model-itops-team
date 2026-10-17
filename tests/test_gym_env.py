"""The environment: determinism, readable errors, the change_ticket rule and seeded chaos."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from gym_env.chaos import ChaosConfig
from gym_env.env import Env
from gym_env.fixtures.make_fixtures import FixturesConfig, generate
from gym_env.state import FIXTURES_DIR, fixture_names, load_fixture

DEMO = "incident_prod_payments"  # payments in prod: revision 9 broken, revision 8 healthy

# A mix of reads and writes across both APIs, including failures.
SCRIPT: list[tuple[str, dict[str, Any]]] = [
    ("list_devices", {"employee_id": "E1004"}),
    ("create_ticket", {"employee_id": "E1004", "category": "hardware", "summary": "Fan noise",
                       "priority": "P2", "device_id": "D2001"}),
    ("create_ticket", {"employee_id": "E1012", "category": "incident", "summary": "Payments down",
                       "priority": "P1"}),
    ("update_ticket", {"ticket_id": "HD-1035", "status": "in_progress", "note": "Looking into it"}),
    ("get_pods", {"namespace": "prod", "deployment": "payments"}),
    ("rollback", {"namespace": "prod", "deployment": "payments", "to_revision": 8}),
    ("rollback", {"namespace": "prod", "deployment": "payments", "to_revision": 8,
                  "change_ticket": "HD-1043"}),
    ("get_pods", {"namespace": "prod", "deployment": "payments"}),
    ("scale", {"namespace": "staging", "deployment": "search", "replicas": 2}),
    ("reset_password", {"employee_id": "E1005", "delivery": "sms"}),
]


def run(env: Env, script: list[tuple[str, dict[str, Any]]] = SCRIPT) -> list[tuple[int, dict[str, Any]]]:
    return [(r.status, r.output) for r in (env.call(name, args) for name, args in script)]


def error(result: Any) -> dict[str, Any]:
    assert not result.ok, result.output
    return result.output["error"]


# Determinism and isolation


def test_same_calls_give_same_results_and_state() -> None:
    first, second = Env(DEMO), Env(DEMO)
    assert run(first) == run(second)
    assert first.state == second.state


def test_ids_come_from_fixture_counters() -> None:
    env = Env("default")
    ids = [
        env.call("create_ticket", {"employee_id": "E1001", "category": "access",
                                   "summary": f"Request {i}", "priority": "P3"}).output["ticket"]["id"]
        for i in range(2)
    ]
    assert ids == ["HD-1042", "HD-1043"]


def test_clock_is_frozen_at_fixture_now() -> None:
    env = Env("default")
    ticket = env.call("create_ticket", {"employee_id": "E1001", "category": "access",
                                        "summary": "VPN access", "priority": "P3"}).output["ticket"]
    assert ticket["created_at"] == ticket["updated_at"] == "2026-10-19T09:00:00Z"


def test_each_task_gets_a_fresh_copy() -> None:
    env = Env(DEMO)
    run(env)
    assert Env(DEMO).state == load_fixture(DEMO)
    assert load_fixture(DEMO).counters == {"ticket": 1042}


def test_pods_follow_the_current_revision() -> None:
    env = Env(DEMO)
    broken = env.call("get_pods", {"namespace": "prod", "deployment": "payments"}).output["pods"]
    assert {p["status"] for p in broken} == {"CrashLoopBackOff"}
    logs = env.call("get_logs", {"namespace": "prod", "pod": broken[0]["name"]}).output["lines"]
    assert any(" WARN " in line or " ERROR " in line for line in logs)

    env.call("create_ticket", {"employee_id": "E1012", "category": "incident",
                               "summary": "Payments down", "priority": "P1"})
    env.call("rollback", {"namespace": "prod", "deployment": "payments", "to_revision": 8,
                          "change_ticket": "HD-1042"})
    healthy = env.call("get_pods", {"namespace": "prod", "deployment": "payments"}).output["pods"]
    assert {p["status"] for p in healthy} == {"Running"}
    assert not {p["name"] for p in healthy} & {p["name"] for p in broken}  # new pod names
    stale = env.call("get_logs", {"namespace": "prod", "pod": broken[0]["name"]})
    assert error(stale)["code"] == "not_found"
    assert "get_pods" in error(stale)["message"]


def test_fixtures_are_reproducible(tmp_path: Path) -> None:
    config = FixturesConfig.model_validate(yaml.safe_load(Path("configs/fixtures.yaml").read_text()))
    for path in generate(config, tmp_path):
        assert json.loads(path.read_text()) == json.loads((FIXTURES_DIR / path.name).read_text()), path.name


def test_every_fixture_loads_and_has_logs_for_every_image() -> None:
    logs = json.loads((FIXTURES_DIR / "logs.json").read_text())["logs"]
    assert "default" in fixture_names() and DEMO in fixture_names()
    for name in fixture_names():
        state = load_fixture(name)
        for deployments in state.namespaces.values():
            for revision in (r for d in deployments for r in d.revisions):
                assert logs[revision.image], revision.image


# Readable errors for invalid arguments


@pytest.mark.parametrize(
    ("tool", "args", "code", "mentions"),
    [
        ("create_ticket", {"employee_id": "E1001", "category": "access", "summary": "x", "priority": "high"},
         "invalid_arguments", ["priority", "'P1', 'P2', 'P3' or 'P4'", "'high'"]),
        ("create_ticket", {"employee_id": "E1001", "category": "access", "priority": "P3"},
         "invalid_arguments", ["missing required argument 'summary'", "create_ticket(employee_id"]),
        ("create_ticket", {"employee_id": "E1001", "category": "access", "summary": "x", "priority": "P3",
                           "severity": 1}, "invalid_arguments", ["unknown argument 'severity'"]),
        ("scale", {"namespace": "staging", "deployment": "search", "replicas": "3"},
         "invalid_arguments", ["replicas", "valid integer"]),
        ("scale", {"namespace": "dev", "deployment": "search", "replicas": 3},
         "invalid_arguments", ["namespace", "'prod' or 'staging'"]),
        ("get_ticket", ["HD-1035"], "invalid_arguments", ["JSON object"]),
        ("get_ticket", {"ticket_id": "HD-9999"}, "not_found", ["HD-9999", "list_tickets"]),
        ("list_devices", {"employee_id": "E9999"}, "not_found", ["E9999"]),
        ("create_ticket", {"employee_id": "E1004", "category": "hardware", "summary": "Fan", "priority": "P2"},
         "failed_precondition", ["device_id", "list_devices(employee_id='E1004')"]),
        ("create_ticket", {"employee_id": "E1004", "category": "hardware", "summary": "Fan", "priority": "P2",
                           "device_id": "D2002"}, "failed_precondition", ["D2002", "E1009", "list_devices"]),
        ("update_ticket", {"ticket_id": "HD-1035"}, "invalid_arguments", ["status, priority or note"]),
        ("get_pods", {"namespace": "prod", "deployment": "payment"}, "not_found", ["payments, checkout"]),
        ("rollback", {"namespace": "staging", "deployment": "search", "to_revision": 7},
         "not_found", ["no revision 7", "30, 31", "rollout_history"]),
        ("delete_ticket", {"ticket_id": "HD-1035"}, "unknown_tool", ["delete_ticket", "create_ticket"]),
    ],
)
def test_invalid_calls_return_readable_errors(tool: str, args: Any, code: str, mentions: list[str]) -> None:
    env = Env("default")
    before = env.state.model_copy(deep=True)
    err = error(env.call(tool, args))
    assert err["code"] == code
    for text in mentions:
        assert text in err["message"]
    assert env.state == before  # a failed call changes nothing, including the ID counters


def test_tools_outside_the_task_apis_are_unknown() -> None:
    env = Env("default", apis=["helpdesk"])
    err = error(env.call("get_pods", {"namespace": "prod", "deployment": "payments"}))
    assert err["code"] == "unknown_tool"
    assert "list_devices" in err["message"]


# The change_ticket rule


@pytest.mark.parametrize(
    ("call", "args"),
    [
        ("rollback", {"namespace": "prod", "deployment": "payments", "to_revision": 8}),
        ("scale", {"namespace": "prod", "deployment": "payments", "replicas": 6}),
    ],
)
def test_prod_change_needs_an_open_ticket(call: str, args: dict[str, Any]) -> None:
    env = Env(DEMO)
    before = env.state.model_copy(deep=True)

    missing = error(env.call(call, args))
    assert missing["code"] == "change_ticket_required"
    assert "prod" in missing["message"] and "change_ticket" in missing["message"]

    for ticket, reason in [("HD-1999", "does not exist"), ("HD-1037", "is resolved")]:
        bad = error(env.call(call, args | {"change_ticket": ticket}))
        assert bad["code"] == "invalid_change_ticket"
        assert reason in bad["message"]
    assert env.state == before

    for ticket in ("HD-1041", "HD-1034"):  # open, in_progress
        assert Env(DEMO).call(call, args | {"change_ticket": ticket}).ok
    assert env.call(call, args | {"change_ticket": "HD-1035"}).ok
    (change,) = env.state.changes
    assert (change.namespace, change.change_ticket) == ("prod", "HD-1035")


def test_ticket_opened_in_the_same_task_authorizes_the_change() -> None:
    env = Env(DEMO)
    ticket = env.call("create_ticket", {"employee_id": "E1012", "category": "incident",
                                        "summary": "Payments down", "priority": "P1"}).output["ticket"]["id"]
    result = env.call("rollback", {"namespace": "prod", "deployment": "payments", "to_revision": 8,
                                   "change_ticket": ticket})
    assert result.ok and result.output["change_ticket"] == "HD-1042"


def test_the_ticket_fixture_adds_change_tickets_to_its_base() -> None:
    base, tickets = load_fixture("incident_prod_checkout"), load_fixture("incident_prod_checkout_tickets")
    assert tickets.namespaces == base.namespaces and tickets.tickets[: len(base.tickets)] == base.tickets
    added = tickets.tickets[len(base.tickets):]
    assert [t.id for t in added] == ["HD-1042", "HD-1043", "HD-1044", "HD-1045", "HD-1046"]
    assert tickets.counters == {"ticket": 1047}
    assert sum(t.employee_id == "E1012" for t in tickets.tickets) == 5  # several tickets per user
    env = Env("incident_prod_checkout_tickets")
    for replicas, ticket, ok in ((3, "HD-1046", True), (4, "HD-1044", True), (5, "HD-1045", False)):  # open, in_progress, resolved
        result = env.call("scale", {"namespace": "prod", "deployment": "search", "replicas": replicas,
                                    "change_ticket": ticket})
        assert result.ok == ok, result.output
    assert env.call("create_ticket", {"employee_id": "E1012", "category": "incident", "summary": "Search slow",
                                      "priority": "P3"}).output["ticket"]["id"] == "HD-1047"


def test_staging_changes_need_no_ticket() -> None:
    env = Env("incident_staging_inventory")
    assert env.call("rollback", {"namespace": "staging", "deployment": "inventory", "to_revision": 9}).ok
    assert env.call("scale", {"namespace": "staging", "deployment": "inventory", "replicas": 2}).ok
    assert [c.change_ticket for c in env.state.changes] == [None, None]


def test_noop_change_is_not_recorded() -> None:
    env = Env(DEMO)
    result = env.call("rollback", {"namespace": "prod", "deployment": "payments", "to_revision": 9})
    assert result.ok and result.output["changed"] is False
    assert env.state == load_fixture(DEMO)


# Seeded chaos

CALL = ("list_deployments", {"namespace": "prod"})


def pattern(chaos: ChaosConfig | None, key: str = "case-1", calls: int = 200) -> list[int]:
    env = Env("default", chaos=chaos, chaos_key=key)
    return [env.call(*CALL).status for _ in range(calls)]


def test_chaos_is_off_by_default() -> None:
    assert ChaosConfig().enabled is False
    assert set(pattern(None)) == {200}


def test_chaos_is_seeded() -> None:
    on = ChaosConfig(enabled=True, rate=0.2, seed=7)
    first = pattern(on)
    assert first == pattern(on)  # same seed and case: same calls fail
    assert set(first) == {200, 503}
    assert 20 <= first.count(503) <= 60  # about 20% of 200
    assert first != pattern(on.model_copy(update={"seed": 8}))
    assert first != pattern(on, key="case-2")


def test_chaos_errors_are_retryable_and_change_nothing() -> None:
    env = Env("default", chaos=ChaosConfig(enabled=True, rate=1.0), chaos_key="x")
    result = env.call("create_ticket", {"employee_id": "E1001", "category": "access",
                                        "summary": "VPN access", "priority": "P3"})
    assert result.status == 503
    assert result.output["error"]["code"] == "service_unavailable"
    assert result.output["error"]["retryable"] is True
    assert "retry" in result.output["error"]["message"]
    assert env.state == load_fixture("default")
