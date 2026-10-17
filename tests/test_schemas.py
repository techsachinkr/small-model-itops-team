"""Each data format loads a valid example and rejects inconsistent ones."""

from typing import Any

import pytest
from pydantic import ValidationError

from common.schemas import Case, Clusters, RunSummary, TrainingExample, Trace

CASE: dict[str, Any] = {
    "id": "p-hd-001",
    "request": "My laptop screen is flickering, please open a ticket.",
    "user_id": "E1001",
    "fixture": "default",
    "apis": ["helpdesk"],
    "gold_calls": [
        {"api": "helpdesk", "tool": "list_devices", "args": {"employee_id": "E1001"}},
        {
            "api": "helpdesk",
            "tool": "create_ticket",
            "args": {
                "employee_id": "E1001",
                "category": "hardware",
                "summary": "Laptop screen flickering",
                "priority": "P3",
                "device_id": "D2001",
            },
        },
    ],
    "kind": "single",
}


def test_case_round_trip() -> None:
    case = Case.model_validate(CASE)
    assert Case.model_validate_json(case.model_dump_json()) == case


@pytest.mark.parametrize(
    "changes",
    [
        {"unknown_field": 1},
        {"kind": "no_tool"},  # no_tool cases have no gold calls
        {"kind": "multi"},  # multi cases need two apis
        {"kind": "read_only"},  # read_only cases need answer_must_contain
        {"apis": ["ops"]},  # gold call api not listed in apis
        {"apis": []},
        # a forbidden string inside a required one makes the case impossible to pass
        {"answer_must_contain": ["ThinkPad X1 Carbon"], "answer_must_not_contain": ["x1 carbon"]},
    ],
)
def test_case_rejects(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Case.model_validate(CASE | changes)


def test_training_example() -> None:
    example = {
        "id": "ex-1",
        "api": "helpdesk",
        "round": 1,
        "kind": "single",
        "origin": "sampled",
        "source": "broad",
        "case_id": "g-hd-001",
        "teacher": "lightning",
        "reward": 1.0,
        "messages": [{"role": "user", "content": "hi"}],
        "text": "<rendered>",
    }
    TrainingExample.model_validate(example)
    with pytest.raises(ValidationError):
        TrainingExample.model_validate(example | {"source": "targeted"})


def test_trace() -> None:
    trace = {
        "run_id": "run-1",
        "setup": "team_trained",
        "case_id": "p-multi-001",
        "attempt": 1,
        "steps": [
            {"agent": "helpdesk", "model": "student", "calls": [], "seatbelt_events": []}
        ],
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "latency_s": 1.5,
        "cost_usd": 0.0001,
        "passed": False,
        "failure_label": "routing",
    }
    Trace.model_validate(trace)
    with pytest.raises(ValidationError):
        Trace.model_validate(trace | {"passed": True})
    with pytest.raises(ValidationError):
        Trace.model_validate(trace | {"setup": "not_a_setup"})


def test_run_summary_fractions() -> None:
    row = {"passes": "7/12", "mean_latency_s": 2.0, "cost_per_1k_usd": 0.5}
    RunSummary.model_validate({"base_seatbelt": row})
    for bad in ("13/12", "58%", "0.58"):
        with pytest.raises(ValidationError):
            RunSummary.model_validate({"base_seatbelt": row | {"passes": bad}})


def test_clusters() -> None:
    cluster = {
        "name": "invented revision",
        "description": "Rolls back to a revision it never looked up.",
        "case_ids": ["p-ops-004"],
        "guidance": "Tasks where rollout_history must be read before rollback.",
    }
    Clusters.model_validate([cluster])
    with pytest.raises(ValidationError):
        Clusters.model_validate([cluster | {"case_ids": []}])
