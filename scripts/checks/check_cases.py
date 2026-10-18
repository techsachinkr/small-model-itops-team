"""Checks drafted cases before they go into data/test/ (TEST_SET_GUIDE.md), as tests/test_practice_cases.py checks the
practice cases: each case parses (common/schemas.py), its gold calls run on its fixture, a gold run passes the
verifier with an answer made of its answer_must_contain strings, doing nothing fails exactly the cases that ask for a
change, and a read_only case's answer strings are in what its gold calls return. Also: unique IDs that aren't
practice IDs, files that match the layout (helpdesk.jsonl and ops.jsonl single-API, any other file multi-API), and
the mix of kinds.

    .venv/bin/python scripts/checks/check_cases.py ~/test_drafts/*.jsonl

It refuses files under data/test/: only eval/run_eval.py --final and eval/decontaminate.py --against test read those
(hard rule 4), so check drafts first, then move them in. Its output quotes the cases: keep it to yourself (Claude must
not see test cases).
"""

import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from pydantic import ValidationError  # noqa: E402

from common.schemas import Case  # noqa: E402
from gym_env.env import Env  # noqa: E402
from gym_env.tasks import read_cases  # noqa: E402
from gym_env.verify import GoldCallFailed, check, expected_state  # noqa: E402

TEST_DIR = REPO / "data" / "test"


def problems_of(case: Case) -> list[str]:
    try:
        expected = expected_state(case)
    except GoldCallFailed as error:
        return [str(error)]
    found = []
    answer = " ".join(case.answer_must_contain or [])
    if not check(case, expected, answer).passed:
        found.append("a gold run doesn't pass the verifier (with the answer_must_contain strings as the answer)")
    idle = check(case, Env(case.fixture, apis=case.apis).state, answer)
    if idle.passed != (case.kind not in ("single", "multi")):
        found.append("doing nothing passes, so the case asks for no change: make it no_tool, missing_info or read_only"
                     if idle.passed else f"doing nothing fails a {case.kind} case: {idle.state_diff[:2]}")
    if case.kind == "read_only":
        env = Env(case.fixture, apis=case.apis)
        returned = " ".join(str(env.call(c.tool, c.args).output) for c in case.gold_calls).casefold()
        missing = [text for text in case.answer_must_contain or [] if text.casefold() not in returned]
        if missing:
            found.append(f"answer_must_contain strings the gold calls never return: {missing}")
    return found


def file_problem(path: Path, case: Case) -> str | None:
    if path.stem in ("helpdesk", "ops") and case.apis != [path.stem]:
        return f"{path.name} holds single-API {path.stem} cases; this one has apis {case.apis}"
    if path.stem not in ("helpdesk", "ops") and len(case.apis) < 2:
        return f"{path.name} would be read for multi-API cases only; a single-API case goes in {case.apis[0]}.jsonl"
    return None


def main() -> None:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        raise SystemExit(__doc__)
    if any(path.resolve().is_relative_to(TEST_DIR) for path in paths):
        raise SystemExit("data/test/ is locked (hard rule 4): check your drafts outside it, then move them in.")
    practice_ids = {case.id for p in sorted((REPO / "data" / "practice").glob("*.jsonl")) for case in read_cases(p)}
    seen: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    bad = 0
    for path in paths:
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                case = Case.model_validate_json(line)
            except ValidationError as error:
                bad += 1
                print(f"{path.name}:{number}: doesn't parse: {error.errors()[0]['msg']} ({error.errors()[0]['loc']})")
                continue
            seen[case.id] += 1
            kinds[case.kind] += 1
            found = problems_of(case)
            if problem := file_problem(path, case):
                found.append(problem)
            if case.id in practice_ids:
                found.append("the ID is a practice case's")
            if seen[case.id] > 1:
                found.append("the ID is used twice")
            bad += bool(found)
            print(f"{path.name}:{number} {case.id} ({case.kind}): {'ok' if not found else 'PROBLEM: ' + '; '.join(found)}")
    total = sum(kinds.values())
    quiet = kinds["no_tool"] + kinds["missing_info"]
    print(f"\n{total} cases, {bad} with problems. Kinds: {dict(kinds)}; no_tool and missing_info {quiet}/{total} "
          f"(aim for about 20%).")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
