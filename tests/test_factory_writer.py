"""factory/writer.py with common/llm.py stubbed: slots, the phrasing and writing checks with their retries,
and the files it writes. No model involved."""

import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from common.llm import LLMResponse
from factory import writer
from factory.sampler import load_factory_config, sample

CONFIG = load_factory_config()
TASKS, WRITER = CONFIG["tasks"], CONFIG["writer"]


def answers(*contents: dict[str, Any]) -> Iterator[LLMResponse]:
    for content in contents:
        yield LLMResponse(message={"role": "assistant", "content": json.dumps(content)}, finish_reason="stop",
                          prompt_tokens=10, completion_tokens=5, latency_s=0.0)


class FakeChat:
    """Stands in for llm.chat: records each call and answers from `reply(...)` in order."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies: Iterator[LLMResponse] = iter(())

    def reply(self, *contents: dict[str, Any]) -> None:
        self.replies = answers(*contents)

    def __call__(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"model": model, "messages": [dict(m) for m in messages], **kwargs})
        return next(self.replies)


@pytest.fixture
def chat(monkeypatch: pytest.MonkeyPatch) -> FakeChat:
    fake = FakeChat()
    monkeypatch.setattr(writer.llm, "chat", fake)
    return fake


def test_slots_mix_origins_reproducibly() -> None:
    slots = writer.slots("helpdesk", 50, "pilot", 0, TASKS)
    assert [s.case_id for s in slots] == [f"hd-pilot-{i:04d}" for i in range(1, 51)]
    assert sum(s.origin == "written" for s in slots) == 10
    assert all((s.sampled is None) == (s.origin == "written") for s in slots)
    again = writer.slots("helpdesk", 50, "pilot", 0, TASKS)
    assert [(s.origin, s.sampled) for s in slots] == [(s.origin, s.sampled) for s in again]
    assert slots[3].rng.random() == again[3].rng.random()


def sampled_slot(template: str) -> writer.Slot:
    drawn = next(t for t in sample("helpdesk", 300, random.Random(0), TASKS["sampled_kinds"]) if t.template == template)
    return writer.Slot("hd-t-0001", "sampled", random.Random(1), drawn)


def test_phrase_retries_with_the_problems_until_the_checks_pass(chat: FakeChat) -> None:
    slot = sampled_slot("hardware_ticket_lookup_device")
    priority = slot.sampled.gold_calls[1].args["priority"]
    chat.reply({"request": "My laptop is broken."}, {"request": f"My laptop is broken, open a {priority} ticket please."})
    outcome = writer.phrase(slot, WRITER)
    assert outcome.case is not None and outcome.case.request.endswith("please.")
    assert outcome.case.gold_calls == slot.sampled.gold_calls and outcome.info["tries"] == 2
    assert "must include the priority" in chat.calls[1]["messages"][-1]["content"]
    assert chat.calls[0]["thinking"] is False and chat.calls[0]["json_schema"] == writer.PHRASE_SCHEMA


def test_phrase_drops_the_task_after_max_tries(chat: FakeChat) -> None:
    slot = sampled_slot("hardware_ticket_lookup_device")
    device_id = slot.sampled.gold_calls[1].args["device_id"]
    chat.reply(*[{"request": f"Laptop {device_id} is broken."}] * WRITER["max_tries"])
    outcome = writer.phrase(slot, WRITER)
    assert outcome.case is None and len(chat.calls) == WRITER["max_tries"]
    assert any("leave out any device ID" in p for p in outcome.problems)


def written(request: str, *calls: tuple[str, dict[str, Any]], answer: list[str] | None = None, kind: str = "single",
            user: str = "E1011") -> tuple[list[str], Any]:
    tools = {"list_devices", "create_ticket", "list_tickets", "get_ticket", "update_ticket", "reset_password"}
    content = {"request": request, "gold_calls": [{"tool": t, "args": a} for t, a in calls], "answer_must_contain": answer or []}
    return writer.written_case(content, "hd-w-0001", "helpdesk", kind, "default", user, tools)


def test_written_case_accepts_a_good_task() -> None:
    problems, case = written("Reset E1003's password and email it to her.",
                             ("reset_password", {"employee_id": "E1003", "delivery": "email"}))
    assert problems == [] and case.kind == "single" and case.gold_calls[0].api == "helpdesk"


@pytest.mark.parametrize(("request_text", "calls", "expected"), [
    ("Reset E1003's password.", [("reset_password", {"employee_id": "E1003", "delivery": "email"})],
     "sets delivery method email"),
    ("Reset Sofia's password by email.", [("reset_password", {"employee_id": "E1003", "delivery": "email"})],
     "colleague's employee ID E1003"),
    ("Mark my ticket HD-1041 done.", [("update_ticket", {"ticket_id": "HD-1041", "status": "resolved", "priority": "P1"})],
     "sets priority P1"),
    ("Note on Marcus's dock ticket: still broken.", [("update_ticket", {"ticket_id": "HD-1034", "note": "Still broken"})],
     "HD-1034, a ticket of E1002"),
    ("Use list_devices for E1003.", [("list_devices", {"employee_id": "E1003"})], "names tools"),
    ("Close HD-9999.", [("update_ticket", {"ticket_id": "HD-9999", "status": "resolved"})], "failed"),
])
def test_written_case_rejects(request_text: str, calls: list[tuple[str, dict[str, Any]]], expected: str) -> None:
    problems, _ = written(request_text, *calls)
    assert any(expected in p for p in problems), problems


def test_written_read_only_answers_come_from_the_lookups() -> None:
    lookup = ("list_devices", {"employee_id": "E1001"})
    assert written("Which laptop model do I have?", lookup, answer=["X1 Carbon"], kind="read_only", user="E1001")[0] == []
    problems, _ = written("Which laptop model do I have?", lookup, answer=["MacBook"], kind="read_only", user="E1001")
    assert "no gold call returns" in problems[0]


def test_write_retries_with_the_problems_then_keeps_the_task(chat: FakeChat) -> None:
    good = {"request": "Please reset E1003's password and text her the new one.",
            "gold_calls": [{"tool": "reset_password", "args": {"employee_id": "E1003", "delivery": "sms"}}],
            "answer_must_contain": []}
    chat.reply({**good, "gold_calls": [{"tool": "reset_password", "args": {"employee_id": "E9999", "delivery": "sms"}}]},
               {**good, "gold_calls": []}, good)
    outcome = writer.write(writer.Slot("hd-w-0001", "written", random.Random(0)), "helpdesk", {"single": 1.0}, WRITER)
    assert outcome.case is not None and outcome.case.request == good["request"] and outcome.info["tries"] == 3
    assert chat.calls[0]["thinking"] is True and chat.calls[0]["json_schema"] == writer.WRITE_SCHEMA
    feedback = [m["content"] for m in chat.calls[2]["messages"] if m["role"] == "user"][1:]
    assert "E9999" in feedback[0] and "single cases need gold_calls" in feedback[1]


def test_main_writes_the_tasks_and_their_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def produce(slot: writer.Slot, api: str, tasks: dict[str, Any], config: dict[str, Any]) -> writer.Outcome:
        if slot.origin == "written":
            return writer.Outcome(slot, None, {"origin": "written", "tries": 3}, ["the gold calls change nothing"])
        return writer.Outcome(slot, slot.sampled.case(slot.case_id, "a request"), {"origin": "sampled", "tries": 1},
                              prompt_tokens=7, completion_tokens=3)

    monkeypatch.setattr(writer, "produce", produce)
    monkeypatch.setattr(sys, "argv", ["writer", "--api", "ops", "--n", "10", "--name", "t", "--out-dir", str(tmp_path)])
    writer.main()
    tasks = [json.loads(line) for line in (tmp_path / "tasks_t.jsonl").read_text().splitlines()]
    config = json.loads((tmp_path / "tasks_t.config.json").read_text())
    assert len(tasks) == 8 and set(config["tasks"]) == {t["id"] for t in tasks}
    assert config["counts"]["kept"] == {"sampled": 8} and config["counts"]["dropped"] == {"written": 2}
    assert config["seed"] == TASKS["seed"] and config["factory_config"] == CONFIG and "git_commit" in config
    assert [d["problems"] for d in config["dropped"]] == [["the gold calls change nothing"]] * 2
    with pytest.raises(SystemExit, match="exists"):
        writer.main()


def test_main_refuses_the_test_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["writer", "--api", "ops", "--n", "1", "--name", "t", "--out-dir", "data/test/x"])
    with pytest.raises(SystemExit, match="locked"):
        writer.main()


def test_naming_email_as_the_problem_is_not_asking_for_email_delivery() -> None:
    # Pilot hd-pilot-0017: Lightning (rightly) asked how to send it, and the rollout failed the state match.
    problems, _ = written("Hi, I need to reset the password for employee E1003. They're locked out and can't access their email.",
                          ("reset_password", {"employee_id": "E1003", "delivery": "email"}))
    assert any("delivery method email" in p for p in problems)
