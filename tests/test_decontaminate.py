"""eval/decontaminate.py against the practice cases: paraphrases and the same changes are dropped, other
tasks are kept. Against the test cases (task 5.1, hard rule 4) it reports and changes nothing; these tests point it
at a stand-in test folder of their own, never data/test/."""

import json
import sys
from pathlib import Path

import pytest

from common.schemas import Case
from eval import decontaminate
from eval.decontaminate import practice_cases, similarity, verdict
from factory.sampler import load_factory_config

PRACTICE = practice_cases()
BY_ID = {case.id: case for case in PRACTICE}
THRESHOLDS = load_factory_config()["decontaminate"]


def task(request: str, user_id: str = "E1009", fixture: str = "default", api: str = "helpdesk", kind: str = "single",
         calls: list[tuple[str, dict]] | None = None) -> Case:
    calls = calls if calls is not None else [("reset_password", {"employee_id": user_id, "delivery": "email"})]
    return Case(id="t-1", request=request, user_id=user_id, fixture=fixture, apis=[api], kind=kind,
                gold_calls=[{"api": api, "tool": tool, "args": args} for tool, args in calls])


def test_a_paraphrase_of_a_practice_request_is_dropped_for_its_text() -> None:
    paraphrase = task("I'm locked out of everything, even my email. Please reset my password and send the temp one "
                      "somewhere I can actually receive it.", user_id="E1003")
    result = verdict(paraphrase, PRACTICE, THRESHOLDS)
    assert result["reason"] == "text" and result["nearest"] == "pr-hd-04"


def test_an_unrelated_request_is_kept() -> None:
    result = verdict(task("Please reset my password and text me the temporary one."), PRACTICE, THRESHOLDS)
    assert result["reason"] is None and result["same_changes_as"] == []


def test_the_same_changes_as_a_practice_case_are_dropped_whatever_the_wording() -> None:
    # pr-ops-05 rolls payments in prod back to revision 6 with change ticket HD-1039.
    rollback = ("rollback", {"namespace": "prod", "deployment": "payments", "to_revision": 6, "change_ticket": "HD-1036"})
    different = task("Payments needs to go back to the build from October 8th, use HD-1036 as the change ticket.",
                     user_id="E1002", api="ops", calls=[("rollout_history", {"namespace": "prod", "deployment": "payments"}), rollback])
    result = verdict(different, PRACTICE, THRESHOLDS)
    assert result["reason"] == "changes" and result["same_changes_as"] == ["pr-ops-05"]


def test_part_of_a_practice_case_counts_as_the_same_changes() -> None:
    # pr-hd-01 resolves HD-1040 and opens a hardware ticket; resolving HD-1040 alone is part of it.
    close = task("All good with the noisy fan now, please close it.", user_id="E1001",
                 calls=[("update_ticket", {"ticket_id": "HD-1040", "status": "resolved", "note": "Quiet now"})])
    assert verdict(close, PRACTICE, THRESHOLDS)["same_changes_as"] == ["pr-hd-01"]


def test_a_near_copy_of_a_case_without_changes_is_dropped_for_the_same_user() -> None:
    # pr-hd-10: E1004, who has no tickets, asks to add a note to "my open ticket".
    copy = task("Can you add a note to my open ticket saying I'm away until Thursday?", user_id="E1004",
                kind="missing_info", calls=[])
    assert verdict(copy, PRACTICE, THRESHOLDS)["reason"] == "same_user"
    other_user = copy.model_copy(update={"user_id": "E1008"})
    assert verdict(other_user, PRACTICE, THRESHOLDS)["reason"] is None


def test_the_same_user_rule_leaves_tasks_with_changes_to_the_other_rules() -> None:
    # E1007 also wrote pr-hd-04 (reset by SMS); a reset by email is a different change.
    reset = task("Can you reset my password and send the temporary one to my email?", user_id="E1007")
    assert verdict(reset, PRACTICE, THRESHOLDS)["reason"] is None


def test_similarity_is_symmetric_and_bounded() -> None:
    a, b = BY_ID["pr-ops-05"].request, BY_ID["pr-ops-06"].request
    assert similarity(a, b) == pytest.approx(similarity(b, a))
    assert similarity(a, a) == pytest.approx(1.0)
    assert 0 <= similarity(a, "Open a P3 ticket for my monitor.") < THRESHOLDS["same_user_threshold"]


def test_practice_cases_are_distinct_under_the_text_threshold() -> None:
    for case in PRACTICE:
        others = [other for other in PRACTICE if other.id != case.id]
        assert max(similarity(case.request, o.request) for o in others) < THRESHOLDS["text_threshold"], case.id


def test_main_rewrites_the_tasks_and_reports_the_drops(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "tasks_t.jsonl"
    kept = task("Please reset my password and text me the temporary one.")
    dropped = BY_ID["pr-hd-04"].model_copy(update={"id": "t-2"})
    path.write_text(kept.model_dump_json() + "\n" + dropped.model_dump_json() + "\n")
    monkeypatch.setattr(sys, "argv", ["decontaminate", "--tasks", str(path), "--against", "practice"])
    decontaminate.main()
    assert [json.loads(line)["id"] for line in path.read_text().splitlines()] == ["t-1"]
    report = json.loads((tmp_path / "tasks_t.decontaminate.json").read_text())
    assert report["kept"] == 1 and report["dropped"][0]["id"] == "t-2" and report["dropped"][0]["task"]["request"]
    with pytest.raises(SystemExit, match="already"):
        decontaminate.main()


@pytest.fixture
def stand_in_test_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A test folder of our own with one case, a copy of pr-hd-04 under a test-looking ID."""
    folder = tmp_path / "test_set"
    folder.mkdir()
    (folder / "helpdesk.jsonl").write_text(BY_ID["pr-hd-04"].model_copy(update={"id": "te-secret-01"}).model_dump_json() + "\n")
    monkeypatch.setattr(decontaminate, "TEST_DIR", folder)
    monkeypatch.setitem(decontaminate.REFERENCES, "test", folder)
    return folder


def test_against_the_test_set_it_reports_without_naming_test_cases_and_changes_nothing(
    stand_in_test_set: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    path = generated / "tasks_round_1.jsonl"
    kept = task("Please reset my password and text me the temporary one.")
    close = BY_ID["pr-hd-04"].model_copy(update={"id": "t-2"})
    original = kept.model_dump_json() + "\n" + close.model_dump_json() + "\n"
    path.write_text(original)
    (generated / "round_1.jsonl").write_text(json.dumps({"case_id": "t-2"}) + "\n" + json.dumps({"case_id": "t-1"}) + "\n")
    monkeypatch.setattr(sys, "argv", ["decontaminate", "--against", "test", "--tasks", str(path)])
    decontaminate.main()

    assert path.read_text() == original  # no file changed: dropping now means retraining, the human's call
    report_text = (generated / "tasks_round_1.decontaminate_test.json").read_text()
    report = json.loads(report_text)
    assert report["test_cases_compared"] == 1 and report["too_close"] == 1
    assert [t["id"] for t in report["too_close_tasks"]] == ["t-2"]
    assert report["training_examples_built_from_them"] == {"round_1.jsonl": 1}
    assert "te-secret-01" not in report_text and "te-secret-01" not in capsys.readouterr().out
    assert not (generated / "tasks_round_1.decontaminate.json").exists()  # the practice report is another file
    with pytest.raises(SystemExit, match="already decontaminated against test"):
        decontaminate.main()


def test_tasks_inside_the_test_folder_are_refused(stand_in_test_set: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for against in ("practice", "test"):
        monkeypatch.setattr(sys, "argv", ["decontaminate", "--against", against, "--tasks", str(stand_in_test_set / "helpdesk.jsonl")])
        with pytest.raises(SystemExit, match="locked"):
            decontaminate.main()


def test_with_no_test_cases_found_it_stops_and_says_where_it_looked(
    stand_in_test_set: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (stand_in_test_set / "helpdesk.jsonl").unlink()  # an empty folder: nothing to compare with
    path = tmp_path / "tasks_t.jsonl"
    path.write_text(task("Please reset my password.").model_dump_json() + "\n")
    monkeypatch.setattr(sys, "argv", ["decontaminate", "--against", "test", "--tasks", str(path)])
    with pytest.raises(SystemExit, match="No test cases found: .* data/test/\\*.jsonl"):
        decontaminate.main()
    assert not (tmp_path / "tasks_t.decontaminate_test.json").exists()
