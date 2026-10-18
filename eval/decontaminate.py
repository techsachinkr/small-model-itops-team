"""Drops generated tasks that are too similar to the practice cases, and reports the ones too similar to the test cases.

    uv run python -m eval.decontaminate --tasks data/generated/helpdesk/pilot/tasks_pilot.jsonl --against practice
    uv run python -m eval.decontaminate --against test --tasks data/generated/*/tasks_round_*.jsonl   # task 5.1, a human

A task is dropped for the first of these that applies (thresholds in configs/factory.yaml):
- text: its request scores at least `text_threshold` against a practice request. The score is the larger
  of two cosine similarities on lowercased text: of word 1- and 2-gram counts, and of character 4-gram
  counts (which catches rewordings like "crashes"/"crashing").
- same_user: it changes nothing (missing_info, no_tool, read_only) and scores at least `same_user_threshold`
  against a practice case of the same kind from the same user on the same fixture: the same situation,
  which the changes check can't see.
- changes: its state-changing gold calls are the same as a practice case's, or a part of them, leaving out
  free text (summaries, notes) and the change ticket. Training on that would teach that case's answer.
Text alone can't tell a paraphrase from a different request built on the same template, so its threshold is
high, and the other two catch what it misses.

--against practice (task 2.1) rewrites each --tasks file with the kept tasks (the file's line order is kept), and
writes <tasks stem>.decontaminate.json next to it: the settings, the practice cases compared, the score distribution,
and every dropped task with its reason, nearest practice case and score.

--against test (task 5.1 step 3) is the only other reader of data/test/ besides eval/run_eval.py --final, and only a
human runs it (hard rule 4). It changes no file: by then the adapters were trained on these tasks, so dropping any
means retraining, which is the human's call. It writes <tasks stem>.decontaminate_test.json with how many test cases
were compared (never which), and each task too close to one with its reason and score, and counts the training
examples built from those tasks in the round files next to them. No test case is named in the report or the output:
the reports sit next to the data we publish.
"""

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.schemas import Case
from eval.run_eval import REPO
from factory.sampler import FACTORY_CONFIG, load_factory_config
from gym_env.tasks import read_cases

CHANGING_TOOLS = {"create_ticket", "update_ticket", "reset_password", "rollback", "scale"}
IGNORED_ARGS = {"summary", "note", "change_ticket"}  # free text, and a ticket the request just names
TEST_DIR = REPO / "data" / "test"
REFERENCES = {"practice": REPO / "data" / "practice", "test": TEST_DIR}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.casefold())


def features(text: str) -> tuple[Counter[str], Counter[str]]:
    tokens = words(text)
    grams = Counter(tokens) + Counter(" ".join(pair) for pair in zip(tokens, tokens[1:]))
    squeezed = " ".join(tokens)
    chars = Counter(squeezed[i:i + 4] for i in range(len(squeezed) - 3))
    return grams, chars


def cosine(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(count * b[key] for key, count in a.items())
    norm = math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def similarity(a: str, b: str) -> float:
    (grams_a, chars_a), (grams_b, chars_b) = features(a), features(b)
    return max(cosine(grams_a, grams_b), cosine(chars_a, chars_b))


def changes(case: Case) -> Counter[str]:
    """The case's state-changing calls without IGNORED_ARGS, as a multiset."""
    return Counter(
        json.dumps({"tool": call.tool, "args": {k: v for k, v in call.args.items() if k not in IGNORED_ARGS}}, sort_keys=True)
        for call in case.gold_calls if call.tool in CHANGING_TOOLS
    )


def verdict(task: Case, references: list[Case], thresholds: dict[str, float]) -> dict[str, Any]:
    """The nearest reference by text, and why the task is dropped (reason None if it's kept)."""
    scored = [(similarity(task.request, ref.request), ref) for ref in references]
    score, nearest = max(scored, key=lambda pair: pair[0])
    situation = (task.kind, task.user_id, task.fixture)
    same_user = max((s for s, ref in scored if (ref.kind, ref.user_id, ref.fixture) == situation
                     and task.kind not in ("single", "multi")), default=0.0)
    task_changes = changes(task)
    same = [ref.id for ref in references if task_changes and not task_changes - changes(ref)]
    if score >= thresholds["text_threshold"]:
        reason = "text"
    elif same_user >= thresholds["same_user_threshold"]:
        reason = "same_user"
    else:
        reason = "changes" if same else None
    return {"id": task.id, "reason": reason, "score": round(score, 3), "nearest": nearest.id,
            "same_user_score": round(same_user, 3), "same_changes_as": same}


def reference_cases(against: str) -> list[Case]:
    """The cases to compare with: data/practice/, or data/test/ (only for --against test: hard rule 4)."""
    return [case for path in sorted(REFERENCES[against].glob("*.jsonl")) for case in read_cases(path)]


def practice_cases() -> list[Case]:
    return reference_cases("practice")


def score_quantiles(verdicts: list[dict[str, Any]]) -> dict[float, float]:
    scores = sorted(v["score"] for v in verdicts)
    return {q: scores[min(len(scores) - 1, int(q * len(scores)))] for q in (0.5, 0.9, 0.99)} if scores else {}


def check_against_test(path: Path, report_path: Path, references: list[Case], thresholds: dict[str, float]) -> None:
    """Reports the tasks too close to a test case, without naming test cases, and changes no file."""
    tasks = read_cases(path)
    verdicts = [verdict(task, references, thresholds) for task in tasks]
    too_close = [{"id": v["id"], "reason": v["reason"], "score": v["score"], "same_user_score": v["same_user_score"],
                  "same_changes_as": len(v["same_changes_as"])} for v in verdicts if v["reason"] is not None]
    ids = {t["id"] for t in too_close}
    examples = {}
    for rounds in sorted(path.parent.glob("round_*.jsonl")):
        built = sum(json.loads(line)["case_id"] in ids for line in rounds.read_text().splitlines() if line.strip())
        examples[rounds.name] = built
    report_path.write_text(json.dumps({
        "script": "eval/decontaminate.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "against": "test",
        "thresholds": thresholds,
        "test_cases_compared": len(references),  # how many, never which (hard rule 4)
        "tasks": len(tasks),
        "too_close": len(too_close),
        "too_close_by_reason": dict(Counter(t["reason"] for t in too_close)),
        "score_quantiles": score_quantiles(verdicts),
        "too_close_tasks": too_close,
        "training_examples_built_from_them": examples,
    }, indent=1))
    print(f"{path}: {len(too_close)}/{len(tasks)} tasks too close to one of the {len(references)} test cases "
          f"({dict(Counter(t['reason'] for t in too_close)) or 'none'}); training examples built from them: {examples}. "
          f"No file changed. Report in {report_path}")
    for t in too_close:
        print(f"  {t['id']} ({t['reason']}): score {t['score']}, same user {t['same_user_score']}, "
              f"same changes as {t['same_changes_as']} test case(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop generated tasks too similar to the practice cases, or report "
                                                 "the ones too similar to the test cases.")
    parser.add_argument("--tasks", type=Path, nargs="+", required=True, help="tasks_<name>.jsonl files")
    parser.add_argument("--against", required=True, choices=["practice", "test"],
                        help="practice: drop and rewrite; test: report only, run by a human (hard rule 4, task 5.1)")
    parser.add_argument("--config", type=Path, default=FACTORY_CONFIG)
    args = parser.parse_args()
    if any(path.resolve().is_relative_to(TEST_DIR) for path in args.tasks):
        raise SystemExit("data/test/ is locked (hard rule 4).")
    suffix = ".decontaminate.json" if args.against == "practice" else ".decontaminate_test.json"
    reports = [path.with_name(path.stem + suffix) for path in args.tasks]
    if done := [str(report) for report in reports if report.exists()]:
        raise SystemExit(f"{', '.join(done)} exist: these tasks were already decontaminated against {args.against}.")
    thresholds = load_factory_config(args.config)["decontaminate"]
    references = reference_cases(args.against)
    for path, report_path in zip(args.tasks, reports):
        if args.against == "test":
            check_against_test(path, report_path, references, thresholds)
        else:
            drop_against_practice(path, report_path, references, thresholds)


def drop_against_practice(path: Path, report_path: Path, references: list[Case], thresholds: dict[str, float]) -> None:
    tasks = read_cases(path)
    verdicts = [verdict(task, references, thresholds) for task in tasks]
    kept = [task for task, v in zip(tasks, verdicts) if v["reason"] is None]
    dropped = [{**v, "task": task.model_dump(mode="json", exclude_none=True)}
               for task, v in zip(tasks, verdicts) if v["reason"] is not None]
    report_path.write_text(json.dumps({
        "script": "eval/decontaminate.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "against": "practice",
        "thresholds": thresholds,
        "references": [case.id for case in references],
        "tasks": len(tasks),
        "kept": len(kept),
        "dropped_by_reason": dict(Counter(d["reason"] for d in dropped)),
        "score_quantiles": score_quantiles(verdicts),
        "dropped": dropped,
    }, indent=1))
    path.write_text("".join(task.model_dump_json(exclude_none=True) + "\n" for task in kept))

    print(f"{path}: kept {len(kept)}/{len(tasks)} against {len(references)} practice cases ({thresholds})")
    for d in dropped:
        print(f"  dropped {d['id']} ({d['reason']}): score {d['score']} vs {d['nearest']}, same user "
              f"{d['same_user_score']}, same changes as {d['same_changes_as']}")
    print(f"Report in {report_path}")


if __name__ == "__main__":
    main()
