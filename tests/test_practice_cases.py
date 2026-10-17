"""Every practice case is well formed: its gold calls run, a gold run passes the verifier, doing nothing
fails exactly the cases that ask for a change, and each read_only answer string is in what the gold
calls return. Reads data/practice/ only (never data/test/, hard rule 4)."""

from pathlib import Path

import pytest

from common.schemas import Case
from gym_env.env import Env
from gym_env.tasks import read_cases
from gym_env.verify import check, expected_state

CASES = [case for path in sorted(Path("data/practice").glob("*.jsonl")) for case in read_cases(path)]


def test_ids_are_unique() -> None:
    ids = [case.id for case in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_case_is_well_formed(case: Case) -> None:
    env = Env(case.fixture, apis=case.apis)
    results = [env.call(call.tool, call.args) for call in case.gold_calls]
    assert all(result.ok for result in results), [r.output for r in results if not r.ok]

    answer = " ".join(case.answer_must_contain or [])
    assert check(case, expected_state(case), answer).passed
    idle = check(case, Env(case.fixture, apis=case.apis).state, answer)
    assert idle.passed == (case.kind not in ("single", "multi")), idle.state_diff

    if case.kind == "read_only":
        returned = " ".join(str(result.output) for result in results).casefold()
        assert all(text.casefold() in returned for text in case.answer_must_contain)


# The wrong behaviours the task 1.2 review found these cases couldn't catch, and the right ones.
BY_ID = {case.id: case for case in CASES}


def run(case_id: str, calls: list[tuple[str, dict]], answer: str = "") -> bool:
    case = BY_ID[case_id]
    env = Env(case.fixture, apis=case.apis)
    for tool, args in calls:
        env.call(tool, args)
    return check(case, env.state, answer).passed


def test_ops_11_guessing_either_namespace_fails() -> None:
    for namespace in ("prod", "staging"):
        assert not run("pr-ops-11", [("scale", {"namespace": namespace, "deployment": "payments", "replicas": 3,
                                                "change_ticket": "HD-1041"})])
    assert run("pr-ops-11", [], "Should I scale payments in prod or in staging?")


def test_ops_12_needs_yesterdays_revision_not_one_step_back() -> None:
    history = [("rollout_history", {"namespace": "prod", "deployment": d})
               for d in ("payments", "checkout", "inventory", "search", "notifications")]
    one_step_back = {"payments": 7, "checkout": 13, "inventory": 4, "search": 21, "notifications": 9}
    naive = [("rollback", {"namespace": "prod", "deployment": d, "to_revision": r, "change_ticket": "HD-1041"})
             for d, r in one_step_back.items()]
    assert not run("pr-ops-12", history + naive)
    # Rolling every deployment to the revision it ran yesterday: four no-ops and checkout 14 -> 13.
    yesterday = {"payments": 8, "checkout": 13, "inventory": 5, "search": 22, "notifications": 10}
    careful = [("rollback", {"namespace": "prod", "deployment": d, "to_revision": r, "change_ticket": "HD-1041"})
               for d, r in yesterday.items()]
    assert run("pr-ops-12", history + careful)


def test_hd_08_hedged_answer_naming_both_devices_fails() -> None:
    lookups = [("list_devices", {"employee_id": "E1001"})]
    assert not run("pr-hd-08", lookups, "Your devices are a Dell U2723QE monitor and a ThinkPad X1 Carbon laptop.")
    assert run("pr-hd-08", lookups, "Your open ticket HD-1040 is about your Lenovo ThinkPad X1 Carbon.")


def test_extra_note_passes_on_the_ticket_in_question_but_not_an_improvised_one() -> None:
    hd01 = [(c.tool, c.args) for c in BY_ID["pr-hd-01"].gold_calls]
    assert run("pr-hd-01", hd01 + [("update_ticket", {"ticket_id": "HD-1042", "note": "Related to HD-1040"})])
    assert not run("pr-hd-12", [("update_ticket", {"ticket_id": "HD-1033", "note": "Please delete"})])
    # Fewer notes than the gold still fails: the dock update without its note.
    assert not run("pr-hd-06", [("update_ticket", {"ticket_id": "HD-1034", "priority": "P2"})])
