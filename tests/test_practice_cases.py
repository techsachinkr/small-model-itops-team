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
