"""Verifier: compares the final state with the state produced by a case's gold calls.

This is the only scoring logic (CLAUDE.md "Scoring"), following NeMo Gym's Execution and State Match
pattern. The expected state comes from running the case's gold calls on a fresh copy of its fixture.
The actual state is the final state of the session the system ran in, which was also a fresh copy.
Both are projected and must match exactly. The projection keeps every structured field, but free text
(ticket summaries and notes) only has to be present, because no model will reproduce the gold wording
(PLAN.md task 1.1, option B). Tickets the gold calls create or update need at least the gold's number of
notes, so a helpful extra note on the ticket the request is about passes; every other ticket must keep
its notes exactly (task 1.2 review). Order-free records (changes, password resets) are compared as sorted
lists. The final answer must contain every answer_must_contain string and none of the
answer_must_not_contain strings (substring checks that ignore case, so "open" matches "Status: Open").
"""

import json
from typing import Any

from pydantic import BaseModel

from common.schemas import Case
from gym_env.env import TOOLS, Env
from gym_env.state import State

TEXT = "<text>"


class GoldCallFailed(Exception):
    """A gold call returned an error, so the case itself is wrong."""


class Verdict(BaseModel):
    passed: bool
    state_match: bool
    answer_ok: bool
    state_diff: list[str]
    missing_from_answer: list[str]
    forbidden_in_answer: list[str]


def gold_run(case: Case) -> Env:
    """A fresh fixture after the case's gold calls."""
    env = Env(case.fixture, apis=case.apis)
    for i, call in enumerate(case.gold_calls):
        tool = TOOLS.get(call.tool)
        if tool is None or tool.api != call.api:
            raise GoldCallFailed(f"{case.id}: gold call {i} names {call.api}.{call.tool}, which doesn't exist")
        result = env.call(call.tool, call.args)
        if not result.ok:
            raise GoldCallFailed(f"{case.id}: gold call {i} ({call.tool}) failed: {result.output}")
    return env


def expected_state(case: Case) -> State:
    return gold_run(case).state


def gold_tickets(env: Env) -> set[str]:
    """IDs of the tickets the gold calls created or updated."""
    return {result.output["ticket"]["id"] for result in env.calls if result.tool in ("create_ticket", "update_ticket")}


def project(state: State) -> dict[str, Any]:
    data = state.model_dump(mode="json")
    tickets = {}
    for ticket in data["tickets"]:
        ticket["summary"] = TEXT if ticket["summary"].strip() else ""
        ticket["notes"] = [TEXT] * len(ticket["notes"])
        tickets[ticket["id"]] = ticket
    deployments = {
        f"{namespace}/{d['name']}": d for namespace, ds in data["namespaces"].items() for d in ds
    }
    return {
        "fixture": data["fixture"],
        "now": data["now"],
        "counters": data["counters"],
        "employees": data["employees"],
        "devices": data["devices"],
        "tickets": tickets,
        "password_resets": sorted(data["password_resets"], key=_canonical),
        "deployments": deployments,
        "changes": sorted(data["changes"], key=_canonical),
    }


def _canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True)


def diff(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Human-readable differences, like `tickets.HD-1042.priority: expected 'P1', got 'P2'`."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        lines = []
        for key in sorted(expected.keys() | actual.keys()):
            where = f"{path}.{key}" if path else str(key)
            if key not in actual:
                lines.append(f"{where}: missing (expected {expected[key]!r})")
            elif key not in expected:
                lines.append(f"{where}: unexpected {actual[key]!r}")
            else:
                lines += diff(expected[key], actual[key], where)
        return lines
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return [line for i, (e, a) in enumerate(zip(expected, actual)) for line in diff(e, a, f"{path}[{i}]")]
    return [] if expected == actual else [f"{path}: expected {expected!r}, got {actual!r}"]


def check(case: Case, final_state: State, final_answer: str) -> Verdict:
    gold = gold_run(case)
    expected, actual = project(gold.state), project(final_state)
    for ticket_id in gold_tickets(gold):
        want, got = expected["tickets"].get(ticket_id), actual["tickets"].get(ticket_id)
        if want and got and len(got["notes"]) >= len(want["notes"]):
            got["notes"] = want["notes"]  # extra notes on a ticket the request is about are fine
    state_diff = diff(expected, actual)
    answer = final_answer.casefold()
    missing = [s for s in case.answer_must_contain or [] if s.casefold() not in answer]
    forbidden = [s for s in case.answer_must_not_contain or [] if s.casefold() in answer]
    return Verdict(
        passed=not state_diff and not missing and not forbidden,
        state_match=not state_diff,
        answer_ok=not missing and not forbidden,
        state_diff=state_diff,
        missing_from_answer=missing,
        forbidden_in_answer=forbidden,
    )
