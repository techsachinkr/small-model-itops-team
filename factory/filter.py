"""Keeps rollouts the verifier passes with no failed calls, rendered with the student's chat template.

    uv run --group train python -m factory.filter --run pilot_helpdesk --round 1 \\
        --output data/generated/helpdesk/pilot/round_1.jsonl --show 5

Reads runs/<run_id>/ from factory/collect.py: rollouts.jsonl, and through config.json the task file with
its .config.json (each task's origin; slots the writer dropped) and .decontaminate.json (tasks dropped
before collection). A rollout is kept only if its `reward` is 1 (the verifier's state match and answer
checks passed, and the seatbelt didn't give up), every call returned 200, and the self-check never sent
Lightning back to work (that nudge is a user turn the student shouldn't learn to wait for). Otherwise
its reason is the first of: gave_up, state_mismatch, answer_mismatch, failed_call (status 0 is a call
the seatbelt blocked), self_check_nudge, other. A task with no rollout is `no_rollout`.

Each kept rollout becomes a TrainingExample (common/schemas.py):
- messages: Lightning's whole conversation in chat format, with its reasoning in `reasoning_content`.
- text: the same conversation rendered as the student is evaluated: the student base model's chat
  template (configs/models.yaml, pinned revision) with enable_thinking=False, the reasoning dropped,
  tool-call arguments as objects and the rollout's own tool definitions, as vLLM passes them. In the
  assistant's text, look-alike hyphens and spaces become ASCII: Lightning writes IDs like "HD‑1041" with
  a non-breaking hyphen (82 of its 199 answers in the task 1.2 runs), which the student would learn and
  the verifier's answer checks would then miss.

Writes --output, <output stem>.stats.json next to it (counts by origin and reason, by kind and template),
and runs/<run_id>/filter_rejected.jsonl (each rejected rollout's task, reason and trace).
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from common.llm import MODELS_CONFIG
from common.schemas import Case, TrainingExample
from eval.run_eval import REPO
from gym_env.tasks import read_cases

# Importing nemo_gym sets HF_HOME to Gym's own cache folder unless it is already set. Keep the usual
# Hugging Face cache, where the student's tokenizer is (nemo_gym is imported only in conversation()).
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
REASONS = ("gave_up", "state_mismatch", "answer_mismatch", "failed_call", "self_check_nudge", "other")


def rejection(row: dict[str, Any]) -> str | None:
    """Why a rollout can't be training data (None if it can)."""
    step = row["trace"]["steps"][0]
    events = step["seatbelt_events"]
    if any(event["kind"] == "gave_up" for event in events):
        return "gave_up"
    if not row.get("state_match"):
        return "state_mismatch"
    if not row.get("answer_ok"):
        return "answer_mismatch"
    if any(call["status"] != 200 for call in step["calls"]):
        return "failed_call"
    if any(event["kind"] == "self_check" and event.get("complete") is False for event in events):
        return "self_check_nudge"
    return None if row.get("reward") == 1.0 else "other"


def conversation(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The rollout's input and output as chat messages (reasoning in reasoning_content), and its tools."""
    from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.responses_converter import ResponsesConverter

    params = dict(row["responses_create_params"])
    params["input"] = [*params["input"], *row["response"]["output"]]
    converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=True)
    chat = converter.responses_to_chat_completion_create_params(NeMoGymResponseCreateParamsNonStreaming.model_validate(params))
    messages = []
    for message in chat.messages:
        message = dict(message)
        if message["role"] == "assistant":
            content = message.get("content") or ""
            thoughts = [t.strip() for t in THINK_BLOCK.findall(content)]
            message["content"] = THINK_BLOCK.sub("", content).strip()
            if any(thoughts):
                message["reasoning_content"] = "\n\n".join(t for t in thoughts if t)
        messages.append(message)
    return messages, [dict(tool) for tool in chat.tools or []]


ASCII_LOOKALIKES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u00a0": " ", "\u202f": " "})


def for_template(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages as the student sees them: no reasoning, ASCII hyphens and spaces in the assistant's text,
    tool-call arguments as objects (as vLLM passes them)."""
    plain = []
    for message in messages:
        message = {key: value for key, value in message.items() if key != "reasoning_content"}
        if message["role"] == "assistant" and isinstance(message.get("content"), str):
            message["content"] = message["content"].translate(ASCII_LOOKALIKES)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {**call, "function": {**call["function"], "arguments": json.loads(call["function"]["arguments"] or "{}")}}
                for call in message["tool_calls"]
            ]
        plain.append(message)
    return plain


@cache
def student_tokenizer() -> Any:
    from transformers import AutoTokenizer

    student = yaml.safe_load(MODELS_CONFIG.read_text())["models"]["student"]
    return AutoTokenizer.from_pretrained(student["hf_id"], revision=student["revision"])


def render(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
    text = student_tokenizer().apply_chat_template(for_template(messages), tools=tools, tokenize=False, enable_thinking=False)
    if re.search(r"<think>(?!</think>)", text):
        raise ValueError("reasoning left in the rendered text")
    return text


def teacher() -> str:
    lightning = yaml.safe_load(MODELS_CONFIG.read_text())["models"]["lightning"]
    return f"{lightning['hf_id']}@{lightning['revision']}"


def sidecar(tasks_path: Path, suffix: str) -> dict[str, Any]:
    path = tasks_path.with_name(tasks_path.stem + suffix)
    return json.loads(path.read_text()) if path.exists() else {}


def task_info(case_id: str, writer_config: dict[str, Any]) -> dict[str, Any]:
    info = writer_config.get("tasks", {}).get(case_id)
    if info is None:
        raise SystemExit(f"{case_id} isn't in the writer's config file, so its origin is unknown.")
    return info


def template_of(info: dict[str, Any]) -> str:
    return info.get("template") or f"written:{info.get('focus')}"


def stats(outcomes: list[dict[str, Any]], writer_config: dict[str, Any], decontamination: dict[str, Any]) -> dict[str, Any]:
    """Counts by origin of every slot: dropped by the writer, decontaminated, kept or rejected (with the reason)."""
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for dropped in writer_config.get("dropped", []):
        table[dropped["origin"]]["writer_dropped"] += 1
    for dropped in decontamination.get("dropped", []):
        info = writer_config.get("tasks", {}).get(dropped["id"], {})
        table[info.get("origin", "unknown")][f"decontaminated_{dropped['reason']}"] += 1
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    by_template: dict[str, Counter[str]] = defaultdict(Counter)
    for outcome in outcomes:
        result = outcome["reason"] or "kept"
        table[outcome["origin"]][result] += 1
        by_kind[outcome["kind"]][result] += 1
        by_template[outcome["template"]][result] += 1
    totals: Counter[str] = sum(table.values(), Counter())
    return {"by_origin": {origin: dict(counts) for origin, counts in sorted(table.items())}, "total": dict(totals),
            "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
            "by_template": {k: dict(v) for k, v in sorted(by_template.items())}}


def show(example: TrainingExample, case: Case, info: dict[str, Any]) -> str:
    lines = [f"=== {example.id} ({example.origin}, {example.kind}, {template_of(info)}; user {case.user_id}, {case.fixture})",
             f"USER: {case.request}"]
    for message in example.messages[2:]:
        if message["role"] == "assistant":
            for call in message.get("tool_calls") or []:
                lines.append(f"  CALL {call['function']['name']}({call['function']['arguments']})")
            if message.get("content"):
                lines.append(f"ASSISTANT: {message['content']}")
        elif message["role"] == "tool":
            output = message["content"]
            lines.append(f"  -> {output if len(output) <= 200 else output[:200] + '...'}")
    lines.append("  gold: " + (", ".join(f"{c.tool}({json.dumps(c.args)})" for c in case.gold_calls) or "no calls"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn verified Lightning rollouts into training examples.")
    parser.add_argument("--run", required=True, help="a run id from factory/collect.py")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="e.g. data/generated/<api>/round_<n>.jsonl")
    parser.add_argument("--show", type=int, default=0, help="print this many kept examples (seeded choice)")
    args = parser.parse_args()

    run_dir = REPO / "runs" / args.run
    collect_config = json.loads((run_dir / "config.json").read_text())
    tasks_path = REPO / collect_config["tasks"]
    cases = {case.id: case for case in read_cases(tasks_path)}
    writer_config = sidecar(tasks_path, ".config.json")
    decontamination = sidecar(tasks_path, ".decontaminate.json")
    rollouts_path = run_dir / "rollouts.jsonl"
    rows = [json.loads(line) for line in rollouts_path.read_text().splitlines() if line.strip()] if rollouts_path.exists() else []
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; remove it or pick another --output.")

    rows_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_case[row["task_id"]].append(row)
    outcomes, examples, rejected = [], [], []
    for case_id, case in cases.items():
        info = task_info(case_id, writer_config)
        base = {"case_id": case_id, "origin": info["origin"], "kind": case.kind, "template": template_of(info)}
        if not rows_by_case.get(case_id):
            outcomes.append({**base, "reason": "no_rollout"})
            continue
        for row in sorted(rows_by_case[case_id], key=lambda r: r["_ng_rollout_index"]):
            reason = rejection(row)
            outcomes.append({**base, "reason": reason})
            if reason is not None:
                rejected.append({**base, "reason": reason, "attempt": row["_ng_rollout_index"] + 1,
                                 "state_diff": row.get("state_diff"), "missing_from_answer": row.get("missing_from_answer"),
                                 "forbidden_in_answer": row.get("forbidden_in_answer"), "trace": row["trace"]})
                continue
            messages, tools = conversation(row)
            examples.append(TrainingExample(
                id=f"{case_id}-{row['_ng_rollout_index'] + 1}", api=case.apis[0], round=args.round, kind=case.kind,
                origin=info["origin"], source="broad", case_id=case_id, teacher=teacher(), reward=row["reward"],
                messages=messages, text=render(messages, tools)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(example.model_dump_json(exclude_none=True) + "\n" for example in examples))
    (run_dir / "filter_rejected.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejected))
    summary = stats(outcomes, writer_config, decontamination)
    stats_path = args.output.with_name(args.output.stem + ".stats.json")
    stats_path.write_text(json.dumps({"run": args.run, "tasks": collect_config["tasks"], "round": args.round,
                                      "rollouts": len(rows), "kept": len(examples), **summary}, indent=1))

    columns = ["writer_dropped", *sorted({k for c in summary["by_origin"].values() for k in c if k.startswith("decontaminated")}),
               "no_rollout", *REASONS, "kept"]
    columns = [c for c in columns if summary["total"].get(c)]
    print(f"{args.run}: {len(examples)} kept from {len(rows)} rollouts on {len(cases)} tasks")
    print(f"{'origin':<10}" + "".join(f"{c:>{max(len(c), 5) + 2}}" for c in columns))
    for origin, counts in [*summary["by_origin"].items(), ("total", summary["total"])]:
        print(f"{origin:<10}" + "".join(f"{counts.get(c, 0):>{max(len(c), 5) + 2}}" for c in columns))
    print("by kind: " + "; ".join(f"{kind} {c.get('kept', 0)}/{sum(c.values())} kept" for kind, c in summary["by_kind"].items()))
    print(f"Examples in {args.output}\nStats in {stats_path}\nRejected rollouts in {run_dir / 'filter_rejected.jsonl'}")
    for example in random.Random(0).sample(examples, min(args.show, len(examples))):
        print("\n" + show(example, cases[example.case_id], task_info(example.case_id, writer_config)))


if __name__ == "__main__":
    main()
