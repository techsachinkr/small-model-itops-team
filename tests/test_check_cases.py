"""scripts/checks/check_cases.py on the practice cases (never data/test/, hard rule 4): well-formed cases pass, broken
ones are reported, and files in data/test/ are refused."""

import importlib.util
import sys
from pathlib import Path

import pytest

from gym_env.tasks import read_cases

SPEC = importlib.util.spec_from_file_location("check_cases", Path("scripts/checks/check_cases.py"))
check_cases = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_cases)
PRACTICE = {path.stem: read_cases(path) for path in sorted(Path("data/practice").glob("*.jsonl"))}


def test_well_formed_cases_have_no_problems() -> None:
    for name, cases in PRACTICE.items():
        for case in cases:
            assert check_cases.problems_of(case) == [], case.id
            assert check_cases.file_problem(Path(f"{name}.jsonl"), case) is None, case.id


def test_broken_cases_and_misplaced_ones_are_reported() -> None:
    hardware = next(c for c in PRACTICE["helpdesk"] if any("device_id" in call.args for call in c.gold_calls))
    calls = [call.model_copy(update={"args": {**call.args, "device_id": "D9999"}}) if "device_id" in call.args else call
             for call in hardware.gold_calls]
    assert "failed" in check_cases.problems_of(hardware.model_copy(update={"gold_calls": calls}))[0]
    multi = PRACTICE["multi"][0]
    assert "single-API ops" in check_cases.file_problem(Path("ops.jsonl"), multi)
    assert "multi-API" in check_cases.file_problem(Path("extra.jsonl"), PRACTICE["ops"][0])


def test_files_in_data_test_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["check_cases.py", str(check_cases.TEST_DIR / "helpdesk.jsonl")])
    with pytest.raises(SystemExit, match="locked"):
        check_cases.main()
