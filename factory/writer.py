"""Phrases sampled actions as requests, and writes whole tasks with Lightning.

    uv run python -m factory.writer --api helpdesk --n 50 --name pilot --out-dir data/generated/helpdesk/pilot

About `tasks.written_fraction` of the tasks are written whole by Lightning (origin=written): it gets the
fixture, the tools and a kind (single or read_only), and returns the request with its gold calls. The
rest come from factory/sampler.py (origin=sampled): code picks the action and computes the gold calls,
and Lightning only phrases the request in a style code picks. Every Lightning call goes through
common/llm.py with a JSON schema. A phrased request must pass the sampled task's include/leave-out
checks, and a written task must pass sampler.task_problems (its gold calls run, change the state if it's
single, and so on); either is retried with the problems as feedback, up to `writer.max_tries`, and then
dropped and logged.

Writes, in --out-dir:
    tasks_<name>.jsonl         the tasks (common/schemas.py Case)
    tasks_<name>.config.json   the factory config, seed, models, git commit, counts, and for each task
                               its origin, template or focus, style and tries; dropped slots and why
Each slot's randomness comes from (seed, api, slot), so the tasks don't depend on thread order. With
LLM_MODE=record the Lightning answers are also saved to runs/cache/.
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from common import llm
from common.schemas import Case, GoldCall
from common.tool_loop import chat_tools
from eval.run_eval import git_commit
from factory.sampler import (
    DELIVERY,
    FACTORY_CONFIG,
    FIXTURES,
    OPS_USERS,
    REPO,
    SampledTask,
    employee,
    exact,
    load_factory_config,
    sample,
    task_problems,
)
from gym_env.env import tool_schemas, tools_for
from gym_env.state import State, load_fixture

ID_PREFIX = {"helpdesk": "hd", "ops": "ops"}
TEST_DIR = REPO / "data" / "test"

STYLES = [
    "one short sentence, casual",
    "two or three sentences, polite, with a little context about why it matters",
    "terse, like a quick chat message; lowercase is fine",
    "slightly frustrated, but clear about what they want",
    "formal and precise",
    "hurried: gets straight to the point, no greeting",
    "friendly and chatty, with a greeting",
]

PHRASE_SYSTEM = """You write the messages employees send to their company's IT assistant in a chat window. The assistant can look up and change helpdesk tickets, devices and passwords, and inspect and change the company's deployments in the prod and staging namespaces.

Write one message, in the employee's own voice, as they would type it. Rules:
- Ask for what the brief describes, in your own words; don't copy the brief's wording.
- Include every detail listed under "Must include", written exactly as given (IDs, priorities, numbers, versions, namespaces).
- Leave out everything listed under "Leave out": the assistant has to look it up or ask.
- Don't add requests, IDs, numbers or facts the brief doesn't give.
- Never mention tools, function names, or how the assistant should do it.
- The employee is the sender. The assistant already knows who they are, so they don't introduce themselves, and they never address the assistant by their own name.
Answer with a JSON object: {"request": "<the message>"}."""

PHRASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"request": {"type": "string"}},
    "required": ["request"],
    "additionalProperties": False,
}

WRITE_SYSTEM = """You design tasks for testing an IT operations assistant. A task is one chat message from an employee, plus the exact tool calls a careful assistant makes to handle it (the gold calls). A task is scored by running its gold calls on a fresh copy of the environment and comparing the final state with the one the assistant leaves, so the gold calls must be exactly right: valid arguments, real IDs from the environment, and every change the request asks for, in order.

The assistant's tools (JSON schemas):
{tools}

Rules:
- The request must give everything the assistant can't look up: a colleague's employee ID (the current user's own ID is known), priorities, numbers, and the change ticket for any change in prod.
- Changes in prod need change_ticket: the ID of an open or in-progress helpdesk ticket, stated in the request.
- Put the lookups a careful assistant needs before acting in the gold calls (for example list_devices before a hardware ticket, rollout_history before rolling back to a version).
- Ticket summaries and notes can have any sensible wording.
- Write a realistic message a real employee would send about their own equipment, accounts and tickets, or, for IT and ops staff, about colleagues they support and systems they run. Every fact in it must match the environment: don't invent tickets, devices, problems or history the environment doesn't have.
- Write it in the employee's voice, as they would type it in chat, without mentioning tools.
- {kind_rule}
Answer with a JSON object: {{"request": "...", "gold_calls": [{{"tool": "...", "args": {{...}}}}], "answer_must_contain": [...]}}."""

KIND_RULES = {
    "single": "The request asks for one or more changes (tickets created or updated, passwords reset, deployments "
              "rolled back or scaled). answer_must_contain is [].",
    "read_only": "The request asks a question the assistant answers only by looking things up; nothing changes. The "
                 "gold calls are the lookups needed. answer_must_contain lists one to three short strings the correct "
                 "answer must contain (an ID, a device model, a version, a pod name), copied exactly from what the "
                 "lookups return. None of them may appear in the request.",
}

WRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "request": {"type": "string"},
        "gold_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"tool": {"type": "string"}, "args": {"type": "object"}},
                "required": ["tool", "args"],
                "additionalProperties": False,
            },
        },
        "answer_must_contain": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["request", "gold_calls", "answer_must_contain"],
    "additionalProperties": False,
}

FOCUS = {
    ("helpdesk", "single"): ["create_ticket", "update_ticket", "reset_password"],
    ("helpdesk", "read_only"): ["list_devices", "list_tickets", "get_ticket"],
    ("ops", "single"): ["rollback", "scale"],
    ("ops", "read_only"): ["list_deployments", "get_pods", "get_logs", "rollout_history"],
}
TWISTS = [
    "two related changes or questions in one message",
    "a condition the assistant must check first (do something only if something is true)",
    "things referred to by description rather than by ID",
    "a short, terse message",
    "a sentence of context about why the employee needs it",
    "a request on behalf of a colleague, by their employee ID",
]


@dataclass
class Slot:
    """One task to produce: its ID, origin and randomness."""

    case_id: str
    origin: str
    rng: random.Random
    sampled: SampledTask | None = None


@dataclass
class Outcome:
    slot: Slot
    case: Case | None
    info: dict[str, Any]
    problems: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def ask(messages: list[dict[str, Any]], schema: dict[str, Any], config: dict[str, Any], thinking: bool, max_tokens: int,
        seed: int, outcome: Outcome) -> dict[str, Any] | None:
    response = llm.chat(config["model"], messages, json_schema=schema, thinking=thinking,
                        temperature=config["temperature"], top_p=config["top_p"], max_tokens=max_tokens, seed=seed)
    outcome.prompt_tokens += response.prompt_tokens
    outcome.completion_tokens += response.completion_tokens
    messages.append({"role": "assistant", "content": response.message.get("content") or ""})
    try:
        answer = json.loads(response.message.get("content") or "")
    except json.JSONDecodeError:
        return None
    return answer if isinstance(answer, dict) else None


def retry(messages: list[dict[str, Any]], problems: list[str]) -> None:
    messages.append({"role": "user", "content": "That doesn't work: " + "; ".join(problems) + ". Fix it and answer again."})


def phrase(slot: Slot, config: dict[str, Any]) -> Outcome:
    """A request for a sampled task, checked against its include and leave-out lists."""
    task = slot.sampled
    user = employee(load_fixture(task.fixture), task.user_id)
    style = slot.rng.choice(STYLES)
    lines = [f"Sender: {user.name}, {user.department}.", f"Brief: {task.brief}"]
    if task.include:
        lines.append("Must include:\n" + "\n".join(f"- {m.text}" for m in task.include))
    if task.leave_out:
        lines.append("Leave out:\n" + "\n".join(f"- {m.text}" for m in task.leave_out))
    lines.append(f"Style: {style}")
    messages = [{"role": "system", "content": PHRASE_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]
    outcome = Outcome(slot, None, {"origin": "sampled", "group": task.group, "template": task.template, "style": style})
    for tries in range(1, config["max_tries"] + 1):
        answer = ask(messages, PHRASE_SCHEMA, config, config["thinking_phrase"], config["max_tokens_phrase"],
                     slot.rng.randrange(2**31), outcome)
        request = (answer or {}).get("request", "").strip()
        problems = task.phrasing_problems(request) if request else ["the answer has no request"]
        outcome.info["tries"] = tries
        if not problems:
            outcome.case = task.case(slot.case_id, request)
            return outcome
        outcome.problems = problems
        retry(messages, problems)
    outcome.info["last_request"] = request
    return outcome


def environment(state: State, api: str) -> dict[str, Any]:
    """What a task writer needs to know about the fixture, like a person writing the practice cases."""
    if api == "helpdesk":
        return {
            "employees": [e.model_dump() for e in state.employees],
            "devices": [d.model_dump() for d in state.devices],
            "tickets": [t.model_dump(exclude={"created_at", "updated_at"}) for t in state.tickets],
        }
    return {
        "namespaces": {
            ns: [{"name": d.name, "replicas": d.replicas, "current_revision": d.current_revision,
                  "revisions": [r.model_dump() for r in d.revisions]} for d in deployments]
            for ns, deployments in state.namespaces.items()
        },
        "note": "healthy=false means that revision's pods crash-loop (the assistant only sees this through pods "
                "and logs). Pod names come from get_pods.",
        "open_helpdesk_tickets_usable_as_change_tickets": [t.id for t in state.tickets if t.status != "resolved"],
    }


def write(slot: Slot, api: str, kinds: dict[str, float], config: dict[str, Any]) -> Outcome:
    """A whole task by Lightning, checked by running its gold calls."""
    rng = slot.rng
    kind = rng.choices(list(kinds), list(kinds.values()))[0]
    state = load_fixture(rng.choice(FIXTURES[api]))
    user = rng.choice(state.employees) if api == "helpdesk" else employee(state, rng.choice(OPS_USERS))
    focus, twist = rng.choice(FOCUS[(api, kind)]), rng.choice(TWISTS)
    tools = json.dumps(chat_tools(tool_schemas([api])))
    system = WRITE_SYSTEM.format(tools=tools, kind_rule=KIND_RULES[kind])
    prompt = (f"Environment (fixture {state.fixture}; the current time is {state.now}):\n"
              f"{json.dumps(environment(state, api))}\n\n"
              f"The employee sending the message: {user.name} ({user.id}, {user.department}).\n"
              f"Write a {kind} task that uses {focus}, with {twist}.")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    outcome = Outcome(slot, None, {"origin": "written", "kind": kind, "focus": focus, "twist": twist})
    tool_names = {tool.name for tool in tools_for([api])}
    for tries in range(1, config["max_tries"] + 1):
        answer = ask(messages, WRITE_SCHEMA, config, config["thinking_write"], config["max_tokens_write"],
                     rng.randrange(2**31), outcome)
        outcome.info["tries"] = tries
        problems, case = written_case(answer, slot.case_id, api, kind, state.fixture, user.id, tool_names)
        if case is not None and not problems:
            outcome.case = case
            return outcome
        outcome.problems = problems
        retry(messages, problems)
    outcome.info["last_answer"] = answer
    return outcome


def written_case(answer: dict[str, Any] | None, case_id: str, api: str, kind: str, fixture: str, user_id: str,
                 tool_names: set[str]) -> tuple[list[str], Case | None]:
    if answer is None:
        return ["the answer isn't a JSON object"], None
    request = str(answer.get("request", "")).strip()
    if mentioned := sorted(name for name in tool_names if "_" in name and name in request):  # "scale" is a word
        return [f"the request names tools: {mentioned}"], None
    try:
        case = Case(id=case_id, request=request, user_id=user_id, fixture=fixture, apis=[api], kind=kind,
                    gold_calls=[GoldCall(api=api, tool=c["tool"], args=c["args"]) for c in answer.get("gold_calls", [])],
                    answer_must_contain=answer.get("answer_must_contain") or None)
    except ValidationError as error:
        return [f"invalid task: {'; '.join(e['msg'] for e in error.errors())}"], None
    except (KeyError, TypeError) as error:
        return [f"invalid task: every gold call needs a tool and args ({error!r})"], None
    if not request:
        return ["the request is empty"], None
    return unstated(case) + task_problems(case), case


STATUS_WORDS = {"resolved": r"\b(clos|resolv|fix|done|sorted)", "in_progress": r"in[ _-]?progress|working on|look(ing)? into",
                "open": r"\b(re-?)?open"}


def unstated(case: Case) -> list[str]:
    """Gold-call values the request must state, because the assistant can't look them up or infer them
    (sampled tasks state them by construction, through their include lists)."""
    state = load_fixture(case.fixture)
    problems = []
    for gold in case.gold_calls:
        args = gold.args
        needed: list[tuple[str, Any, str | None]] = [
            ("priority", args.get("priority"), exact(str(args.get("priority")))),
            ("change ticket", args.get("change_ticket"), exact(str(args.get("change_ticket")))),
            ("delivery method", args.get("delivery"), DELIVERY[args["delivery"]].pattern if args.get("delivery") in DELIVERY else None),
        ]
        if args.get("employee_id") not in (None, case.user_id):
            needed.append(("the colleague's employee ID", args["employee_id"], exact(str(args["employee_id"]))))
        if gold.tool == "update_ticket" and args.get("status") is not None:
            needed.append(("status", args["status"], STATUS_WORDS.get(args["status"])))
        ticket_id = args.get("ticket_id")
        owner = next((t.employee_id for t in state.tickets if t.id == ticket_id), None)
        if (owner not in (None, case.user_id) and not re.search(exact(str(ticket_id)), case.request, re.I)
                and not re.search(exact(owner), case.request, re.I)):
            problems.append(f"{gold.tool} uses {ticket_id}, a ticket of {owner}: the assistant can only find it if "
                            f"the request gives the ticket ID or {owner}")
        for what, value, pattern in needed:
            if value is not None and (pattern is None or not re.search(pattern, case.request, re.I)):
                problems.append(f"{gold.tool} sets {what} {value}, but the request doesn't say so")
    return list(dict.fromkeys(problems))


def slots(api: str, n: int, name: str, seed: int, tasks_config: dict[str, Any]) -> list[Slot]:
    """The n slots in ID order, with the sampled tasks drawn up front from one seeded generator."""
    rng = random.Random(seed)
    n_written = round(n * tasks_config["written_fraction"])
    written = set(rng.sample(range(n), n_written))
    sampled = iter(sample(api, n - n_written, rng, tasks_config["sampled_kinds"]))
    return [Slot(case_id=f"{ID_PREFIX[api]}-{name}-{i + 1:04d}", origin="written" if i in written else "sampled",
                 rng=random.Random(f"{seed}:{api}:{i}"), sampled=None if i in written else next(sampled))
            for i in range(n)]


def produce(slot: Slot, api: str, tasks_config: dict[str, Any], writer_config: dict[str, Any]) -> Outcome:
    if slot.origin == "written":
        return write(slot, api, tasks_config["written_kinds"], writer_config)
    return phrase(slot, writer_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tasks: sampled actions phrased by Lightning, plus tasks Lightning writes.")
    parser.add_argument("--api", required=True, choices=["helpdesk", "ops"])
    parser.add_argument("--n", type=int, required=True, help="number of task slots (dropped slots are logged)")
    parser.add_argument("--name", required=True, help="task set name, e.g. pilot or round_1")
    parser.add_argument("--out-dir", type=Path, help="default: data/generated/<api>")
    parser.add_argument("--config", type=Path, default=FACTORY_CONFIG)
    parser.add_argument("--seed", type=int, help="default: the config's tasks.seed")
    args = parser.parse_args()

    out_dir = (args.out_dir or REPO / "data" / "generated" / args.api).resolve()
    if out_dir.is_relative_to(TEST_DIR):
        raise SystemExit("data/test/ is locked (hard rule 4).")
    tasks_path, config_path = out_dir / f"tasks_{args.name}.jsonl", out_dir / f"tasks_{args.name}.config.json"
    if tasks_path.exists():
        raise SystemExit(f"{tasks_path} exists; pick another --name or remove it.")
    factory_config = load_factory_config(args.config)
    tasks_config, writer_config = factory_config["tasks"], factory_config["writer"]
    seed = tasks_config["seed"] if args.seed is None else args.seed

    # The code that runs is the code at the start: a run takes most of an hour, and commits made meanwhile
    # (round 1 recorded the end-of-run commit, which differed from the start only in docs) don't count.
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance = git_commit(out_dir)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    todo = slots(args.api, args.n, args.name, seed, tasks_config)
    print(f"{args.api}: {len(todo)} slots ({sum(s.origin == 'written' for s in todo)} written), seed {seed}")
    with ThreadPoolExecutor(writer_config["concurrency"]) as pool:
        outcomes = list(pool.map(lambda slot: produce(slot, args.api, tasks_config, writer_config), todo))
    elapsed = time.perf_counter() - started

    kept = [o for o in outcomes if o.case is not None]
    tasks_path.write_text("".join(o.case.model_dump_json(exclude_none=True) + "\n" for o in kept))
    models = yaml.safe_load(llm.MODELS_CONFIG.read_text())["models"]
    config_path.write_text(json.dumps({
        "script": "factory/writer.py",
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "started_at": started_at,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **provenance,  # at the start
        "factory_config": factory_config,
        "seed": seed,
        "llm_mode": os.environ.get("LLM_MODE", "live"),
        "writer_model": {key: models[writer_config["model"]][key] for key in ("hf_id", "revision")},
        "counts": {
            "slots": len(outcomes),
            "kept": dict(Counter(o.slot.origin for o in kept)),
            "dropped": dict(Counter(o.slot.origin for o in outcomes if o.case is None)),
            "prompt_tokens": sum(o.prompt_tokens for o in outcomes),
            "completion_tokens": sum(o.completion_tokens for o in outcomes),
            "elapsed_s": round(elapsed, 1),
        },
        "tasks": {o.slot.case_id: o.info for o in kept},
        "dropped": [{"case_id": o.slot.case_id, **o.info, "problems": o.problems} for o in outcomes if o.case is None],
    }, indent=1))
    by_kind = Counter((o.slot.origin, o.case.kind) for o in kept)
    print(f"kept {len(kept)}/{len(outcomes)} in {elapsed:.0f} s: "
          + ", ".join(f"{origin} {kind} {k}" for (origin, kind), k in sorted(by_kind.items())))
    for o in outcomes:
        if o.case is None:
            print(f"  dropped {o.slot.case_id} ({o.slot.origin}): {'; '.join(o.problems)}")
    print(f"Tasks in {tasks_path}\nConfig in {config_path}")


if __name__ == "__main__":
    main()
