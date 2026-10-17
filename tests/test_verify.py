"""The verifier passes the gold calls and fails a wrong final state."""

from pathlib import Path
from typing import Any

import pytest

from common.schemas import Case
from gym_env.env import Env
from gym_env.state import State
from gym_env.tasks import read_cases
from gym_env.verify import GoldCallFailed, check

CASES = {case.id: case for case in read_cases(Path("resources_servers/itops/data/example_cases.jsonl"))}


def final_state(case: Case, calls: list[tuple[str, dict[str, Any]]]) -> State:
    """What a system leaves behind after making these calls on a fresh fixture (errors included)."""
    env = Env(case.fixture, apis=case.apis)
    for name, args in calls:
        env.call(name, args)
    return env.state


def gold(case: Case) -> list[tuple[str, dict[str, Any]]]:
    return [(c.tool, c.args) for c in case.gold_calls]


ANSWERS = {"ex-ro-001": "Your VPN ticket HD-1035 is priority P2 and still open."}


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_gold_calls_pass(case_id: str) -> None:
    case = CASES[case_id]
    verdict = check(case, final_state(case, gold(case)), ANSWERS.get(case_id, "Done."))
    assert verdict.passed, verdict


def test_free_text_only_has_to_be_present() -> None:
    case = CASES["ex-multi-001"]
    calls = gold(case)
    calls[0] = ("create_ticket", calls[0][1] | {"summary": "payments 2.5.0 crash loop, rolled back"})
    assert check(case, final_state(case, calls), "").passed


def test_extra_reads_and_recovered_errors_still_pass() -> None:
    case = CASES["ex-multi-001"]
    rollback = ("rollback", {"namespace": "prod", "deployment": "payments", "to_revision": 8})
    calls = [("get_pods", {"namespace": "prod", "deployment": "payments"}), rollback, *gold(case)]
    assert check(case, final_state(case, calls), "").passed


def _with(case: Case, index: int, **changes: Any) -> list[tuple[str, dict[str, Any]]]:
    calls = gold(case)
    calls[index] = (calls[index][0], calls[index][1] | changes)
    return calls


@pytest.mark.parametrize(
    ("case_id", "calls", "expected_diff"),
    [
        ("ex-hd-001", lambda c: _with(c, 1, priority="P3"), "tickets.HD-1042.priority"),
        ("ex-hd-001", lambda c: _with(c, 1, category="software", device_id=None), "tickets.HD-1042.category"),
        ("ex-hd-001", lambda c: gold(c)[:1], "tickets.HD-1042: missing"),
        ("ex-ops-001", lambda c: _with(c, 0, replicas=2), "deployments.staging/checkout.replicas"),
        ("ex-ops-002", lambda c: _with(c, 1, to_revision=10), "deployments.staging/inventory.current_revision"),
        # Rolled back with an unrelated open ticket instead of the one opened for this incident.
        ("ex-multi-001", lambda c: _with(c, 2, change_ticket="HD-1041"), "change_ticket"),
        ("ex-multi-001", lambda c: gold(c)[:1], "deployments.prod/payments.current_revision"),
        ("ex-mi-001", lambda c: [("scale", {"namespace": "staging", "deployment": "search", "replicas": 3})],
         "deployments.staging/search.replicas"),
        ("ex-nt-001", lambda c: [("reset_password", {"employee_id": "E1003", "delivery": "email"})],
         "password_resets"),
        ("ex-ro-001", lambda c: [("update_ticket", {"ticket_id": "HD-1035", "note": "checked"})],
         "tickets.HD-1035.notes"),
    ],
)
def test_wrong_final_state_fails(case_id: str, calls: Any, expected_diff: str) -> None:
    case = CASES[case_id]
    verdict = check(case, final_state(case, calls(case)), ANSWERS.get(case_id, ""))
    assert not verdict.passed and not verdict.state_match
    assert any(expected_diff in line for line in verdict.state_diff), verdict.state_diff


def test_read_only_answer_must_contain_every_string() -> None:
    case = CASES["ex-ro-001"]
    state = final_state(case, gold(case))
    verdict = check(case, state, "Your VPN ticket is P2.")
    assert verdict.state_match and not verdict.passed
    assert verdict.missing_from_answer == ["open"]


def test_answer_check_ignores_case() -> None:
    case = CASES["ex-ro-001"]  # answer_must_contain: ["P2", "open"]
    state = final_state(case, gold(case))
    assert check(case, state, "Priority: p2\nStatus: **Open**").passed


def test_failing_gold_call_is_a_case_error() -> None:
    case = CASES["ex-hd-001"].model_copy(deep=True)
    case.gold_calls[1].args.pop("device_id")
    with pytest.raises(GoldCallFailed, match="device_id"):
        check(case, final_state(case, []), "")
