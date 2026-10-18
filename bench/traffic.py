"""Real agent traffic for the serving benchmarks (task 4.1): every model call of the specialists' practice runs,
rebuilt as the exact prompt vLLM rendered, with its real output length.

    uv run python -m bench.traffic [--run r1_helpdesk --run r1_ops] [--output-dir runs/bench/traffic]

Each rollout of a run (one specialist on its own adapter, thinking off; by default task 2.2's r1 runs) is one agent
session. factory.filter.conversation() turns it back into chat messages, and each assistant turn, with tool calls or
an answer, was one model call: the conversation before it, rendered with the student's chat template, the run's tools
and enable_thinking=False (the prompt vLLM builds for that chat request), with the turn's own tokens as its output.
After each answer the seatbelt asked its self-check: a call without tools on the conversation plus SELF_CHECK_PROMPT,
whose output is the recorded verdict as compact JSON. Prompts keep the model's own text (no ASCII mapping, unlike
training text).

Writes <output-dir>/<api>.jsonl, one {"prompt", "output_tokens", "session", "kind"} per call with each session's calls
in order, which `vllm bench serve --dataset-name custom --skip-chat-template` reads as it is (bench/run.py), and
traffic.json: the source runs, the git commit, and per API the sessions, the calls by kind, and the rebuilt prompt and
output tokens next to the ones the run recorded.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from common.seatbelt import SELF_CHECK_PROMPT
from eval.run_eval import REPO, git
from factory.filter import conversation, student_tokenizer

RUNS = REPO / "runs"
OUTPUT_DIR = RUNS / "bench" / "traffic"
DEFAULT_RUNS = ("r1_helpdesk", "r1_ops")


def for_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages as vLLM passes them to the chat template: no reasoning, tool-call arguments as objects."""
    plain = []
    for message in messages:
        message = {key: value for key, value in message.items() if key != "reasoning_content"}
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {**call, "function": {**call["function"], "arguments": json.loads(call["function"]["arguments"] or "{}")}}
                for call in message["tool_calls"]
            ]
        plain.append(message)
    return plain


def as_sent(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool definitions as the eval requests reached vLLM: without the converter's `strict: false`, which would add
    7 tokens per tool (with it, every tools call came out 42 tokens longer than the runs recorded)."""
    return [{**tool, "function": {k: v for k, v in tool["function"].items() if k != "strict"}} for tool in tools]


def render(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, generation_prompt: bool) -> str:
    return student_tokenizer().apply_chat_template(for_prompt(messages), tools=as_sent(tools) if tools else None,
                                                   tokenize=False, add_generation_prompt=generation_prompt,
                                                   enable_thinking=False)


def tokens(text: str) -> int:
    return len(student_tokenizer()(text, add_special_tokens=False).input_ids)


def prompt(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    """The prompt vLLM builds for a chat request with these messages and tools (None: no tools), thinking off."""
    return render(messages, tools, generation_prompt=True)


def turn_tokens(before: list[dict[str, Any]], turn: dict[str, Any], tools: list[dict[str, Any]]) -> int:
    """The tokens generated for `turn`: the conversation rendered with it, after the prompt that asked for it."""
    asked, answered = prompt(before, tools), render([*before, turn], tools, generation_prompt=False)
    if not answered.startswith(asked):
        raise ValueError("the rendered turn doesn't continue the prompt that asked for it")
    return tokens(answered) - tokens(asked)


def call(text: str, output_tokens: int, session: str, kind: str) -> dict[str, Any]:
    return {"prompt": text, "output_tokens": output_tokens, "session": session, "kind": kind}


def session_calls(row: dict[str, Any], session: str) -> list[dict[str, Any]]:
    """One rollout's model calls, in the order they were made."""
    messages, tools = conversation(row)
    verdicts = iter([event for event in row["trace"]["steps"][0]["seatbelt_events"] if event["kind"] == "self_check"])
    calls = []
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = messages[:i]
        kind = "tool_call" if message.get("tool_calls") else "answer"
        calls.append(call(prompt(before, tools), turn_tokens(before, message, tools), session, kind))
        if kind == "answer" and (verdict := next(verdicts, None)) is not None:
            check = [*messages[: i + 1], {"role": "user", "content": SELF_CHECK_PROMPT}]
            answer = json.dumps({"reason": verdict["detail"], "tool_call_needed": not verdict["complete"]},
                                separators=(",", ":"))  # compact, as the servers' structured outputs write it
            calls.append(call(prompt(check, None), tokens(answer) + 1, session, "self_check"))  # + the end token
    return calls


def build(run_id: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """A run's API, its calls, and their stats."""
    run_dir = RUNS / run_id
    api = json.loads((run_dir / "config.json").read_text())["args"]["api"]
    rows = [json.loads(line) for line in (run_dir / "rollouts.jsonl").read_text().splitlines() if line.strip()]
    calls = [c for row in rows for c in session_calls(row, f"{row['task_id']}#{row['_ng_rollout_index'] + 1}")]
    stats = {
        "run": run_id,
        "sessions": len(rows),
        "calls": dict(Counter(c["kind"] for c in calls)),
        "prompt_tokens": sum(tokens(c["prompt"]) for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "recorded_prompt_tokens": sum(row["trace"]["prompt_tokens"] for row in rows),
        "recorded_completion_tokens": sum(row["trace"]["completion_tokens"] for row in rows),
    }
    return api, calls, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the specialists' model calls for the serving benchmarks.")
    parser.add_argument("--run", action="append", help=f"a single-API run under runs/; default {' and '.join(DEFAULT_RUNS)}")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"git_commit": git("rev-parse", "HEAD").strip(), "apis": {}}
    for run_id in args.run or DEFAULT_RUNS:
        api, calls, stats = build(run_id)
        (args.output_dir / f"{api}.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
        summary["apis"][api] = stats
        print(f"{api} ({run_id}): {stats['sessions']} sessions, {len(calls)} calls {stats['calls']}; prompt tokens "
              f"{stats['prompt_tokens']:,} (run recorded {stats['recorded_prompt_tokens']:,}), output tokens "
              f"{stats['output_tokens']:,} (recorded {stats['recorded_completion_tokens']:,})")
    (args.output_dir / "traffic.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
