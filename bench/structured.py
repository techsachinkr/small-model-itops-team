"""Serving benchmark (4), task 4.1: how often our structured outputs come back valid with vLLM's guided decoding (the
`response_format` JSON schema every setup sends) and without it.

    uv run python -m bench.structured [--repeats 3] [--run-id <id>]

Two kinds of request, rebuilt as the evals send them, with thinking off:
- plans: the planner's request (agents/orchestrator.py) on each multi-API practice case with both specialists active,
  answered by Lightning (the planner of team_trained) and by the student base (the planner of team_small_planner),
  --repeats times each;
- self-checks: the seatbelt's question after each answer in the r1 practice runs (the sessions bench/traffic.py
  rebuilds), answered by the specialist's own adapter (as in r1 and team_trained) and by Lightning.
Each request goes once with `response_format` (strict, as common/tool_loop.py sends it) and once without; both prompts
already ask for the JSON object. An answer is JSON if the whole text parses, as the planner and the seatbelt parse it,
and fits if it is JSON that the schema accepts (jsonschema). JSON inside a code fence is counted apart: a more
lenient parser could use it, ours doesn't. Sampling as configs/eval.yaml except max_tokens, capped at 2,048 (a
structured answer here is under 600 tokens). Needs both vLLM servers as they run (no restarts); about 2 minutes.

Writes runs/bench/<run_id>/structured.json (each answer's grade, with the text of the ones that don't fit) and the
"4. Structured outputs" section of runs/bench/summary.md (bench/run.py keeps it when it rewrites the file).
"""

import argparse
import asyncio
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from agents.orchestrator import MAX_PLAN_STEPS, PLANNER_PROMPT, plan_schema
from agents.specialist import load_specialists
from common.eval_config import load_eval_config
from common.llm import port_for
from common.seatbelt import SELF_CHECK_PROMPT, SELF_CHECK_SCHEMA
from eval.run_eval import REPO, git_commit, load_cases
from gym_env.tasks import system_prompt

BENCH = REPO / "runs" / "bench"
SUMMARY = BENCH / "summary.md"
RUNS = ("r1_helpdesk", "r1_ops")
CONCURRENCY = 16
MAX_TOKENS = 2048
FENCE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)
SECTION_START, SECTION_END = "<!-- 4. structured outputs -->", "<!-- /4. structured outputs -->"


def plan_requests(repeats: int) -> list[dict[str, Any]]:
    specialists = list(load_specialists().values())
    cards, names = [s.card for s in specialists], [s.name for s in specialists]
    requests = []
    for case in load_cases("practice", None):
        prompt = PLANNER_PROMPT.format(context=system_prompt(case), cards=json.dumps(cards, indent=1), max_steps=MAX_PLAN_STEPS)
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": case.request}]
        requests += [{"kind": "plan", "case": case.id, "messages": messages, "schema": plan_schema(names)}] * repeats
    return requests


def self_check_requests(run_id: str) -> list[dict[str, Any]]:
    """The seatbelt's self-check after every answer of a single-API run, as the tool loop asks it (no tools)."""
    from factory.filter import conversation

    run_dir = REPO / "runs" / run_id
    api = json.loads((run_dir / "config.json").read_text())["args"]["api"]
    requests = []
    for line in (run_dir / "rollouts.jsonl").read_text().splitlines():
        row = json.loads(line)
        messages, _ = conversation(row)
        for i, message in enumerate(messages):
            if message["role"] == "assistant" and not message.get("tool_calls"):
                history = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages[: i + 1]]
                requests.append({"kind": "self_check", "api": api, "case": row["task_id"], "schema": SELF_CHECK_SCHEMA,
                                 "messages": [*history, {"role": "user", "content": SELF_CHECK_PROMPT}]})
    return requests


def schema_errors(text: str, schema: dict[str, Any]) -> list[str]:
    """Why a JSON answer doesn't fit (the first few of the schema's complaints)."""
    errors = Draft202012Validator(schema).iter_errors(json.loads(text.strip()))
    return [f"{'.'.join(map(str, e.absolute_path)) or '(top)'}: {e.message[:120]}" for e in list(errors)[:3]]


def grade(text: str, schema: dict[str, Any]) -> dict[str, Any]:
    """`json`: the whole text parses; `fits`: and the schema accepts it; `fenced`: JSON only inside a code fence."""
    value, parsed = None, "not_json"
    try:
        value, parsed = json.loads(text.strip()), "json"
    except json.JSONDecodeError:
        if match := FENCE.search(text):
            try:
                value, parsed = json.loads(match.group(1)), "fenced"
            except json.JSONDecodeError:
                pass
    fits = value is not None and not any(Draft202012Validator(schema).iter_errors(value))
    return {"json": parsed == "json", "fenced": parsed == "fenced", "fits": fits and parsed == "json",
            "fits_if_unfenced": fits and parsed == "fenced"}


async def answer(client: Any, model: str, request: dict[str, Any], guided: bool, sampling: dict[str, Any],
                 limit: asyncio.Semaphore) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": request["messages"], "temperature": sampling["temperature"],
                            "top_p": sampling["top_p"], "max_tokens": MAX_TOKENS,
                            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    if guided:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "output", "schema": request["schema"], "strict": True}}
    async with limit:
        start = time.perf_counter()
        completion = await client.chat.completions.create(**body)
        latency = time.perf_counter() - start
    text = completion.choices[0].message.content or ""
    graded = grade(text, request["schema"])
    return {"kind": request["kind"], "case": request["case"], "model": model, "guided": guided, **graded,
            "cut_off": completion.choices[0].finish_reason == "length", "latency_s": latency,
            "output_tokens": completion.usage.completion_tokens, "text": None if graded["fits"] else text,
            "errors": None if graded["fits"] or not graded["json"] else schema_errors(text, request["schema"])}


def jobs(plans: list[dict[str, Any]], checks: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], bool]]:
    """(model, request, guided) for every answer to collect."""
    adapters = {"helpdesk": "helpdesk_r1", "ops": "ops_r1"}
    out = []
    for guided in (True, False):
        out += [(model, request, guided) for model in ("lightning", "student") for request in plans]
        out += [(model, request, guided) for request in checks for model in (adapters[request["api"]], "lightning")]
    return out


def table(answers: list[dict[str, Any]]) -> str:
    groups: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for a in answers:
        groups[(a["kind"], "the specialist's adapter" if a["model"].endswith("_r1") else a["model"], a["guided"])].append(a)
    lines = ["| request | model | guided decoding | JSON | fits the schema | JSON only in a code fence | cut off at "
             f"{MAX_TOKENS:,} tokens | mean output tokens | mean latency (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for (kind, model, guided), group in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], not item[0][2])):
        n = len(group)

        def count(key: str) -> str:
            return f"{sum(a[key] for a in group)}/{n}"

        lines.append(f"| {kind.replace('_', '-')} | {model} | {'on' if guided else 'off'} | {count('json')} | {count('fits')} | "
                     f"{count('fenced')} | {count('cut_off')} | {sum(a['output_tokens'] for a in group) / n:.0f} | "
                     f"{sum(a['latency_s'] for a in group) / n:.2f} |")
    return "\n".join(lines)


def section(run_id: str, answers: list[dict[str, Any]], plans: int, checks: int, repeats: int) -> str:
    return "\n".join([
        SECTION_START,
        "## 4. Structured outputs: valid with and without guided decoding",
        "",
        f"Run `{run_id}` (`bench/structured.py`), on the servers as they run. Plans: the planner's request on the "
        f"{plans // repeats} multi-API practice cases with both specialists, {repeats} times each, from Lightning and "
        f"from the student base. Self-checks: the seatbelt's question after each of the {checks} answers in the r1 "
        "practice runs, from the specialist's own adapter and from Lightning. Thinking off, sampling as "
        f"`configs/eval.yaml` but at most {MAX_TOKENS:,} tokens. JSON: the whole answer parses, as our code parses it; "
        "fits: and the JSON schema accepts it. Servers: `scripts/serve_lightning.sh` and `scripts/serve_student.sh` as "
        "they run, both with `--structured-outputs-config '{\"backend\": \"xgrammar\", \"disable_any_whitespace\": "
        "true}'`; guided decoding is `response_format` with the request's JSON schema, `strict`.",
        "",
        table(answers),
        SECTION_END,
    ]) + "\n"


def with_section(text: str, block: str) -> str:
    """The summary with this section in place of an earlier one, or added at the end."""
    if SECTION_START in text and SECTION_END in text:
        before, rest = text.split(SECTION_START, 1)
        return before + block + rest.split(SECTION_END, 1)[1].lstrip("\n")
    return text.rstrip("\n") + "\n\n" + block


async def collect(work: list[tuple[str, dict[str, Any], bool]], sampling: dict[str, Any]) -> list[dict[str, Any]]:
    from openai import AsyncOpenAI

    clients = {port: AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
               for port in {port_for(model) for model, _, _ in work}}
    limit = asyncio.Semaphore(CONCURRENCY)
    return await asyncio.gather(*(answer(clients[port_for(model)], model, request, guided, sampling, limit)
                                  for model, request, guided in work))


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured outputs with and without guided decoding.")
    parser.add_argument("--repeats", type=int, default=3, help="answers per plan request")
    parser.add_argument("--run-id", help="default: <time>_structured")
    args = parser.parse_args()
    plans = plan_requests(args.repeats)
    checks = [request for run_id in RUNS for request in self_check_requests(run_id)]
    work = jobs(plans, checks)
    started = datetime.now(timezone.utc)
    run_id = args.run_id or f"{started:%Y%m%d-%H%M%S}_structured"
    run_dir = BENCH / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    sampling = load_eval_config().sampling()
    print(f"{run_id}: {len(work)} answers ({len(plans)} plan and {len(checks)} self-check requests, each by 2 models, "
          "with and without guided decoding)")
    answers = asyncio.run(collect(work, sampling))
    (run_dir / "structured.json").write_text(json.dumps({
        "run_id": run_id, "started_at": started.isoformat(), "max_tokens": MAX_TOKENS, "sampling": sampling,
        "repeats": args.repeats, **git_commit(run_dir), "answers": answers}, indent=2))
    block = section(run_id, answers, len(plans), len(checks), args.repeats)
    SUMMARY.write_text(with_section(SUMMARY.read_text() if SUMMARY.exists() else "# Serving benchmarks (task 4.1)\n", block))
    print(table(answers))
    print(f"Results in {run_dir}; section 4 of {SUMMARY}")


if __name__ == "__main__":
    main()
