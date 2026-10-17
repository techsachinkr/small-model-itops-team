"""factory/filter.py on synthetic Gym rollouts: which rollouts are kept, the chat messages, the student's
rendering (skipped if the student's tokenizer isn't in the local Hugging Face cache), and the files and
stats it writes."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from common.schemas import Case, TrainingExample
from factory import filter as factory_filter
from gym_env.tasks import case_to_row

CASE = Case(id="hd-t-0001", request="Reset my password and text me the temporary one.", user_id="E1007",
            fixture="default", apis=["helpdesk"], kind="single",
            gold_calls=[{"api": "helpdesk", "tool": "reset_password", "args": {"employee_id": "E1007", "delivery": "sms"}}])


def tokenizer_available() -> bool:
    try:
        factory_filter.student_tokenizer()
    except Exception:  # not downloaded: rendering can't be checked here
        return False
    return True


needs_tokenizer = pytest.mark.skipif(not tokenizer_available(), reason="student tokenizer not in the HF cache")


def row(case: Case = CASE, reward: float = 1.0, state_match: bool = True, answer_ok: bool = True,
        statuses: tuple[int, ...] = (200,), events: list[dict[str, Any]] | None = None, attempt: int = 0) -> dict[str, Any]:
    """A rollout shaped like Gym's: the task row, the response's output items, the verdict and our trace."""
    output: list[dict[str, Any]] = [
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "Email is locked, so SMS."}]},
        {"type": "function_call", "call_id": "c1", "id": "c1", "name": "reset_password", "status": "completed",
         "arguments": json.dumps({"employee_id": "E1007", "delivery": "sms"})},
        {"type": "function_call_output", "call_id": "c1", "status": "completed",
         "output": json.dumps({"employee_id": "E1007", "delivery": "sms", "sent_to": "the mobile number on file"})},
        {"type": "message", "id": "m1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "annotations": [], "text": "Done: a temporary password for E1007 went by SMS (see HD‑1041)."}]},
    ]
    calls = [{"turn": 1, "tool": "reset_password", "args": {}, "status": s, "output": {}} for s in statuses]
    return {
        **case_to_row(case),
        "response": {"output": output},
        "reward": reward, "verifier_reward": reward, "state_match": state_match, "answer_ok": answer_ok,
        "state_diff": [], "missing_from_answer": [], "forbidden_in_answer": [], "_ng_rollout_index": attempt,
        "trace": {"steps": [{"agent": "single", "model": "lightning", "calls": calls, "seatbelt_events": events or []}],
                  "prompt_tokens": 100, "completion_tokens": 20, "latency_s": 1.0},
    }


GAVE_UP = {"kind": "gave_up", "turn": 3, "detail": "retry limit"}
NUDGE = {"kind": "self_check", "turn": 1, "detail": "a reset is needed", "complete": False}


@pytest.mark.parametrize(("rollout", "reason"), [
    (row(), None),
    (row(reward=0.0, events=[GAVE_UP]), "gave_up"),
    (row(reward=0.0, state_match=False), "state_mismatch"),
    (row(reward=0.0, answer_ok=False), "answer_mismatch"),
    (row(statuses=(0, 200)), "failed_call"),  # blocked by the seatbelt, then fixed
    (row(statuses=(400, 200)), "failed_call"),
    (row(events=[NUDGE]), "self_check_nudge"),
    (row(reward=0.0, state_match=False, events=[GAVE_UP]), "gave_up"),
])
def test_rejection(rollout: dict[str, Any], reason: str | None) -> None:
    assert factory_filter.rejection(rollout) == reason


def test_conversation_moves_reasoning_out_of_the_content() -> None:
    messages, tools = factory_filter.conversation(row())
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    call = messages[2]
    assert call["reasoning_content"] == "Email is locked, so SMS." and "<think>" not in call["content"]
    assert json.loads(call["tool_calls"][0]["function"]["arguments"]) == {"employee_id": "E1007", "delivery": "sms"}
    assert messages[4]["content"].startswith("Done:") and "reasoning_content" not in messages[4]
    assert [t["function"]["name"] for t in tools][:2] == ["list_devices", "create_ticket"]


@needs_tokenizer
def test_render_is_the_student_view_without_reasoning() -> None:
    messages, tools = factory_filter.conversation(row())
    text = factory_filter.render(messages, tools)
    assert "Email is locked" not in text and text.count("<think></think>") == 2
    assert "<function=reset_password>\n<parameter=employee_id>\nE1007\n</parameter>" in text
    assert "by SMS (see HD-1041)." in text  # look-alike space and hyphen made ASCII
    assert "‑" in messages[4]["content"]  # messages keep Lightning's own text
    student = factory_filter.student_tokenizer()
    prompt = student.apply_chat_template(factory_filter.for_template(messages[:2]), tools=tools, tokenize=False,
                                         enable_thinking=False, add_generation_prompt=True)
    assert text.startswith(prompt)  # each turn renders as vLLM prompts the student, thinking off


@needs_tokenizer
def test_main_writes_examples_and_stats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    second = CASE.model_copy(update={"id": "hd-t-0002"})
    missing = CASE.model_copy(update={"id": "hd-t-0003"})
    tasks = tmp_path / "data" / "tasks_t.jsonl"
    tasks.parent.mkdir()
    tasks.write_text("".join(c.model_dump_json() + "\n" for c in (CASE, second, missing)))
    tasks.with_name("tasks_t.config.json").write_text(json.dumps({
        "tasks": {"hd-t-0001": {"origin": "sampled", "template": "reset_password"},
                  "hd-t-0002": {"origin": "written", "focus": "reset_password"},
                  "hd-t-0003": {"origin": "sampled", "template": "reset_password"},
                  "hd-t-0004": {"origin": "written", "focus": "create_ticket"}},
        "dropped": [{"case_id": "hd-t-0005", "origin": "written", "problems": ["x"]}]}))
    tasks.with_name("tasks_t.decontaminate.json").write_text(json.dumps({"dropped": [{"id": "hd-t-0004", "reason": "text"}]}))
    run_dir = tmp_path / "runs" / "c1"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"tasks": "data/tasks_t.jsonl"}))
    rollouts = [row(), row(second, statuses=(0, 200))]
    (run_dir / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rollouts))
    output = tmp_path / "out" / "round_1.jsonl"
    monkeypatch.setattr(factory_filter, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["filter", "--run", "c1", "--round", "1", "--output", str(output), "--show", "1"])
    factory_filter.main()

    examples = [TrainingExample.model_validate_json(line) for line in output.read_text().splitlines()]
    assert [(e.id, e.origin, e.source, e.round) for e in examples] == [("hd-t-0001-1", "sampled", "broad", 1)]
    assert examples[0].teacher.startswith("nvidia/NVIDIA-Nemotron-3.5-Lightning") and "<tool_call>" in examples[0].text
    stats = json.loads((tmp_path / "out" / "round_1.stats.json").read_text())
    assert stats["by_origin"] == {"sampled": {"kept": 1, "no_rollout": 1},
                                  "written": {"writer_dropped": 1, "decontaminated_text": 1, "failed_call": 1}}
    rejected = [json.loads(line) for line in (run_dir / "filter_rejected.jsonl").read_text().splitlines()]
    assert [(r["case_id"], r["reason"]) for r in rejected] == [("hd-t-0002", "failed_call")]
    assert "=== hd-t-0001-1" in capsys.readouterr().out
