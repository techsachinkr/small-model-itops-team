"""Drops generated tasks that are too similar to the practice cases.

    uv run python -m eval.decontaminate --tasks data/generated/helpdesk/pilot/tasks_pilot.jsonl --against practice

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

Only `--against practice` exists for now (task 2.1). Hard rule 4 lets this script read data/test/ only when
a human asks for that run, so that option needs adding first.

Rewrites --tasks with the kept tasks (the file's line order is kept), and writes
<tasks stem>.decontaminate.json next to it: the settings, the practice cases compared, the score
distribution, and every dropped task with its reason, nearest practice case and score.
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


def practice_cases() -> list[Case]:
    return [case for path in sorted((REPO / "data" / "practice").glob("*.jsonl")) for case in read_cases(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop generated tasks too similar to the practice cases.")
    parser.add_argument("--tasks", type=Path, required=True, help="a tasks_<name>.jsonl file; rewritten in place")
    parser.add_argument("--against", required=True, choices=["practice"], help="practice only for now (hard rule 4)")
    parser.add_argument("--config", type=Path, default=FACTORY_CONFIG)
    args = parser.parse_args()
    if args.tasks.resolve().is_relative_to(TEST_DIR):
        raise SystemExit("data/test/ is locked (hard rule 4).")
    report_path = args.tasks.with_name(args.tasks.stem + ".decontaminate.json")
    if report_path.exists():
        raise SystemExit(f"{report_path} exists: these tasks were already decontaminated.")
    thresholds = load_factory_config(args.config)["decontaminate"]

    references = practice_cases()
    tasks = read_cases(args.tasks)
    verdicts = [verdict(task, references, thresholds) for task in tasks]
    kept = [task for task, v in zip(tasks, verdicts) if v["reason"] is None]
    dropped = [{**v, "task": task.model_dump(mode="json", exclude_none=True)}
               for task, v in zip(tasks, verdicts) if v["reason"] is not None]
    scores = sorted(v["score"] for v in verdicts)
    report_path.write_text(json.dumps({
        "script": "eval/decontaminate.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "against": args.against,
        "thresholds": thresholds,
        "references": [case.id for case in references],
        "tasks": len(tasks),
        "kept": len(kept),
        "dropped_by_reason": dict(Counter(d["reason"] for d in dropped)),
        "score_quantiles": {q: scores[min(len(scores) - 1, int(q * len(scores)))] for q in (0.5, 0.9, 0.99)} if scores else {},
        "dropped": dropped,
    }, indent=1))
    args.tasks.write_text("".join(task.model_dump_json(exclude_none=True) + "\n" for task in kept))

    print(f"{args.tasks}: kept {len(kept)}/{len(tasks)} against {len(references)} practice cases ({thresholds})")
    for d in dropped:
        print(f"  dropped {d['id']} ({d['reason']}): score {d['score']} vs {d['nearest']}, same user "
              f"{d['same_user_score']}, same changes as {d['same_changes_as']}")
    print(f"Report in {report_path}")


if __name__ == "__main__":
    main()
