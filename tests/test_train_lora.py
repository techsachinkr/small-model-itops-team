"""factory/train_lora.py without a GPU: which tokens get a loss (checked against the student's real chat template
and tokenizer, skipped if they aren't in the local Hugging Face cache), the held-out split, and the settings."""

import json
import re
from pathlib import Path

import pytest

from common.schemas import TrainingExample
from eval.run_eval import REPO
from factory import filter as factory_filter
from factory import train_lora
from factory.sampler import load_factory_config


def tokenizer_available() -> bool:
    try:
        factory_filter.student_tokenizer()
    except Exception:  # not downloaded: the template and tokenizer can't be checked here
        return False
    return True


needs_tokenizer = pytest.mark.skipif(not tokenizer_available(), reason="student tokenizer not in the HF cache")


def tool(name: str, properties: dict[str, dict[str, str]]) -> dict:
    return {"type": "function", "function": {"name": name, "description": f"The {name} tool.", "parameters": {
        "type": "object", "properties": properties, "required": list(properties)}}}


def call(call_id: str, name: str, args: dict[str, str]) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


TOOLS = [tool("list_devices", {"employee_id": {"type": "string"}}),
         tool("create_ticket", {"employee_id": {"type": "string"}, "device_id": {"type": "string"}})]
# Three kinds of assistant turn: text with a call, a call alone, and the final answer.
MESSAGES = [
    {"role": "system", "content": "You are an IT operations assistant.\nCurrent user: E1007"},
    {"role": "user", "content": "My laptop screen flickers. Please open a hardware ticket."},
    {"role": "assistant", "content": "Let me find your laptop first.",
     "tool_calls": [call("c1", "list_devices", {"employee_id": "E1007"})]},
    {"role": "tool", "tool_call_id": "c1", "content": json.dumps([{"device_id": "LT-2007", "type": "laptop"}])},
    {"role": "assistant", "content": "", "tool_calls": [call("c2", "create_ticket", {"employee_id": "E1007", "device_id": "LT-2007"})]},
    {"role": "tool", "tool_call_id": "c2", "content": json.dumps({"ticket_id": "HD-1042"})},
    {"role": "assistant", "content": "I opened HD-1042 for your laptop LT-2007."},
]
TURNS = [2, 4, 6]  # index of each assistant message


def labeled_runs(labels: list[int]) -> list[list[int]]:
    """Positions of each run of consecutive tokens that get a loss."""
    runs: list[list[int]] = []
    for position, label in enumerate(labels):
        if label == -100:
            continue
        if runs and runs[-1][-1] == position - 1:
            runs[-1].append(position)
        else:
            runs.append([position])
    return runs


def test_assistant_spans_start_after_the_generation_prompt_and_end_with_the_stop_token() -> None:
    text = ("<|im_start|>system\nYou help.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n<think></think><tool_call>\n<function=list_devices>\n</function>\n</tool_call>\n<|im_end|>\n"
            "<|im_start|>user\n<tool_response>\n[]\n</tool_response>\n<|im_end|>\n"
            "<|im_start|>assistant\n<think></think>You have no devices.<|im_end|>\n")
    assert [text[begin:end] for begin, end in train_lora.assistant_spans(text)] == [
        "<tool_call>\n<function=list_devices>\n</function>\n</tool_call>\n<|im_end|>", "You have no devices.<|im_end|>"]


@pytest.mark.parametrize("text, problem", [
    ("<|im_start|>user\nHi<|im_end|>\n", "no assistant turn"),
    ("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n", "doesn't start with"),
    ("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>Hello", "has no <|im_end|>"),
])
def test_assistant_spans_refuse_text_vllm_wouldn_t_produce(text: str, problem: str) -> None:
    with pytest.raises(ValueError, match=re.escape(problem)):
        train_lora.assistant_spans(text)


@needs_tokenizer
def test_labels_cover_exactly_what_the_student_generates() -> None:
    tokenizer = factory_filter.student_tokenizer()
    text = factory_filter.render(MESSAGES, TOOLS)
    row = train_lora.tokenize(text, tokenizer)
    ids, runs, spans = row["input_ids"], labeled_runs(row["labels"]), train_lora.assistant_spans(text)
    assert ids == tokenizer(text, add_special_tokens=False)["input_ids"]
    assert all(row["labels"][p] == ids[p] for run in runs for p in run)
    assert len(runs) == len(spans) == len(TURNS)
    assert [tokenizer.decode([ids[p] for p in run]) for run in runs] == [text[begin:end] for begin, end in spans]
    assert tokenizer.decode([ids[p] for p in runs[0]]).startswith("Let me find your laptop first.\n<tool_call>")
    assert tokenizer.decode([ids[p] for p in runs[2]]) == "I opened HD-1042 for your laptop LT-2007.<|im_end|>"
    for run, turn in zip(runs, TURNS):
        assert ids[run[-1]] == tokenizer.convert_tokens_to_ids(train_lora.END_OF_TURN)  # the student learns to stop
        # Everything before the turn is exactly the prompt vLLM sends for it, token for token: the conversation
        # so far and the generation prompt, which itself gets no loss.
        prompt = tokenizer.apply_chat_template(factory_filter.for_template(MESSAGES[:turn]), tools=TOOLS, tokenize=False,
                                               enable_thinking=False, add_generation_prompt=True)
        assert prompt.endswith(train_lora.GENERATION_PROMPT)
        assert ids[:run[0]] == tokenizer(prompt, add_special_tokens=False)["input_ids"]


def example(number: int, api: str = "helpdesk") -> TrainingExample:
    case_id = f"hd-t-{number:04d}"
    return TrainingExample(id=f"{case_id}-1", api=api, round=1, kind="single", origin="sampled", source="broad",
                           case_id=case_id, teacher="lightning", reward=1.0, messages=[], text="")


def test_the_held_out_split_is_seeded_and_disjoint() -> None:
    examples = [example(number) for number in range(100)]
    train, held_out = train_lora.split(examples, 0.05, seed=0)
    assert (len(train), len(held_out)) == (95, 5)
    assert {e.id for e in train}.isdisjoint(e.id for e in held_out)
    assert {e.id for e in train_lora.split(examples[::-1], 0.05, seed=0)[1]} == {e.id for e in held_out}  # not order
    assert {e.id for e in train_lora.split(examples, 0.05, seed=1)[1]} != {e.id for e in held_out}
    assert train_lora.split(examples, 0.0, seed=0) == (examples, [])


def test_an_adapter_trains_on_its_own_api_only(tmp_path: Path) -> None:
    path = tmp_path / "round_1.jsonl"
    path.write_text("".join(e.model_dump_json() + "\n" for e in (example(1), example(2, api="ops"))))
    with pytest.raises(SystemExit, match="ops examples"):
        train_lora.load_examples([path], "helpdesk")
    assert train_lora.data_files("ops", 2) == [REPO / "data/generated/ops/round_1.jsonl",
                                                REPO / "data/generated/ops/round_2.jsonl"]


def test_the_shipped_settings_fit_the_student_server() -> None:
    settings = train_lora.TrainSettings.model_validate(load_factory_config()["train"])
    serve = (REPO / "scripts" / "serve_student.sh").read_text()
    assert settings.lora.r <= int(re.search(r"--max-lora-rank (\d+)", serve).group(1))
