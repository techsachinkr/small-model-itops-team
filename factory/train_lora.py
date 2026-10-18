"""LoRA fine-tuning of the student on one API's kept Lightning rollouts (or every API's), with TRL and PEFT.

    uv run --group train python -m factory.train_lora --api helpdesk --round 1
    uv run --group train python -m factory.train_lora --api all --round 1

Trains on data/generated/<api>/round_1.jsonl through round_<n>.jsonl (factory/filter.py; --data overrides) with
the `train:` settings in configs/factory.yaml. `--api all` trains single_trained's adapter, all_r<n>, on every
API's examples with the same settings: each API's examples are held out exactly as for that API's own adapter, and
each trajectory is rendered again with every API's tools, as the single_* setups prompt the model (rendering it
with its own API's tools must give back its stored text, so the tools are the only change). It writes:
    adapters/<api>_r<n>/     the PEFT adapter, which scripts/serve_student.sh serves as model "<api>_r<n>",
                             and training_run.json (which run made it, from which data, rendered how)
    runs/train_<api>_r<n>/   config.json     arguments, settings, base model and revision, data files with their
                                             sha256, how the text was rendered, token counts, held-out example
                                             ids, git commit
                             log.jsonl       the Trainer's logs as they arrive (loss, learning rate, eval loss)
                             metrics.json    final metrics: runtime, losses, peak GPU memory, the full log
                             git_diff.patch  uncommitted changes, when the tree is dirty

The loss covers only what the student generates. Each example's `text` is the conversation rendered with the
student's chat template, thinking off, and vLLM's prompt for every assistant turn ends with GENERATION_PROMPT, so
each turn's loss runs from there through its <|im_end|> (the stop token). System, user and tool-result tokens and
the generation prompt itself get label -100. An example longer than `max_length` stops the run instead of being
cut, which would drop its final answer.

For a dry run that measures memory and speed, cap the steps and examples and write the adapter elsewhere:
    uv run --group train python -m factory.train_lora --api helpdesk --round 1 --max-steps 3 --limit 80 \\
        --run-id dryrun_train_helpdesk --output <dir>
"""

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from common.llm import MODELS_CONFIG
from common.schemas import Strict, TrainingExample
from eval.run_eval import APIS, REPO, git_commit
from factory.filter import render, student_tokenizer
from factory.sampler import FACTORY_CONFIG, load_factory_config
from gym_env.env import tool_schemas

# Importing nemo_gym sets HF_HOME to Gym's own cache folder unless it is already set. Keep the usual Hugging Face
# cache, where the student is; transformers is imported only in train(), after this.
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))

# The student's chat template with enable_thinking=False: vLLM's prompt for an assistant turn ends with this, and
# the template renders every earlier assistant turn the same way (each example has one user request).
GENERATION_PROMPT = "<|im_start|>assistant\n<think></think>"
ASSISTANT_TURN = "<|im_start|>assistant\n"
END_OF_TURN = "<|im_end|>"
EVERY_API = "all"  # --api all: one adapter on every API's examples (single_trained)


class LoraSettings(Strict):
    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0, lt=1)
    target_modules: list[str]


class TrainSettings(Strict):
    """The `train:` section of configs/factory.yaml."""

    seed: int
    lora: LoraSettings
    learning_rate: float = Field(gt=0)
    epochs: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    lr_scheduler: str
    warmup_fraction: float = Field(ge=0, lt=1)
    weight_decay: float = Field(ge=0)
    gradient_checkpointing: bool
    max_length: int = Field(gt=0)
    eval_fraction: float = Field(ge=0, lt=1)
    logging_steps: int = Field(gt=0)


def assistant_spans(text: str) -> list[tuple[int, int]]:
    """Character spans the student generates: each assistant turn after GENERATION_PROMPT, through END_OF_TURN."""
    spans = []
    start = text.find(ASSISTANT_TURN)
    while start != -1:
        if not text.startswith(GENERATION_PROMPT, start):
            raise ValueError(f"the assistant turn at character {start} doesn't start with {GENERATION_PROMPT!r}")
        begin = start + len(GENERATION_PROMPT)
        end = text.find(END_OF_TURN, begin)
        if end == -1:
            raise ValueError(f"the assistant turn at character {start} has no {END_OF_TURN}")
        spans.append((begin, end + len(END_OF_TURN)))
        start = text.find(ASSISTANT_TURN, end)
    if not spans:
        raise ValueError("no assistant turn")
    return spans


def tokenize(text: str, tokenizer: Any) -> dict[str, list[int]]:
    """Token ids for the whole text, with no special tokens added (the chat template has them, and vLLM adds none
    to a rendered chat prompt), and labels that are -100 outside the assistant spans."""
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = assistant_spans(text)
    labels = []
    for token_id, (first, last) in zip(encoding["input_ids"], encoding["offset_mapping"], strict=True):
        if any(first < edge < last for span in spans for edge in span):
            raise ValueError(f"the token at characters {first}-{last} crosses the edge of an assistant turn")
        labels.append(token_id if any(begin <= first < end for begin, end in spans) else -100)
    return {"input_ids": list(encoding["input_ids"]), "labels": labels}


def data_files(api: str, round_: int) -> list[Path]:
    apis = APIS if api == EVERY_API else (api,)
    return [REPO / "data" / "generated" / name / f"round_{k}.jsonl" for name in apis for k in range(1, round_ + 1)]


def load_examples(paths: list[Path], api: str) -> list[TrainingExample]:
    examples = []
    for path in paths:
        examples += [TrainingExample.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    found = {example.api for example in examples}
    if api == EVERY_API:
        if missing := sorted(set(APIS) - found):
            raise SystemExit(f"The data has no {', '.join(missing)} examples; --api all trains on every API's.")
    elif others := sorted(found - {api}):
        raise SystemExit(f"The data has {', '.join(others)} examples; this run trains the {api} adapter only.")
    if len({example.id for example in examples}) != len(examples):
        raise SystemExit("The data has duplicate example ids.")
    return examples


def split(examples: list[TrainingExample], fraction: float, seed: int) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """A seeded split into training examples and examples held out for eval loss, made for each API on its own, so the
    --api all adapter holds out exactly the examples each API's own adapter held out."""
    held_out: set[str] = set()
    for api in sorted({example.api for example in examples}):
        ids = sorted(example.id for example in examples if example.api == api)
        held_out |= set(random.Random(seed).sample(ids, round(len(ids) * fraction)))
    return ([example for example in examples if example.id not in held_out],
            [example for example in examples if example.id in held_out])


def with_every_tool(example: TrainingExample) -> str:
    """The example's trajectory rendered with every API's tools, in the order the single_* setups' rows list them."""
    if render(example.messages, tool_schemas([example.api])) != example.text:
        raise SystemExit(f"{example.id}: its messages don't render to its stored text, so re-rendering them would "
                         "change more than the tools.")
    return render(example.messages, tool_schemas(APIS))


def token_counts(rows: list[dict[str, list[int]]]) -> dict[str, int]:
    lengths = [len(row["input_ids"]) for row in rows]
    return {"examples": len(rows), "tokens": sum(lengths), "max_tokens": max(lengths, default=0),
            "trained_tokens": sum(label != -100 for row in rows for label in row["labels"])}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shown(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return shown(value.resolve())
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def train(settings: TrainSettings, train_rows: list[dict[str, list[int]]], eval_rows: list[dict[str, list[int]]],
          output: Path, run_dir: Path, max_steps: int | None) -> dict[str, Any]:
    """Trains the adapter, saves it to `output`, and returns the final metrics."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class LogToFile(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            with (run_dir / "log.jsonl").open("a") as log:
                log.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **(logs or {})}) + "\n")

    free_at_start, _ = torch.cuda.mem_get_info()
    student = yaml.safe_load(MODELS_CONFIG.read_text())["models"]["student"]
    model = AutoModelForCausalLM.from_pretrained(student["hf_id"], revision=student["revision"], dtype=torch.bfloat16)
    lora = LoraConfig(r=settings.lora.r, lora_alpha=settings.lora.alpha, lora_dropout=settings.lora.dropout,
                      target_modules=settings.lora.target_modules, task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=str(run_dir / "trainer"),
        seed=settings.seed,
        data_seed=settings.seed,
        num_train_epochs=settings.epochs,
        max_steps=max_steps or -1,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        lr_scheduler_type=settings.lr_scheduler,
        warmup_steps=settings.warmup_fraction,  # below 1: a fraction of the optimizer steps
        weight_decay=settings.weight_decay,
        gradient_checkpointing=settings.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_length=settings.max_length,
        eval_strategy="epoch" if eval_rows else "no",
        prediction_loss_only=True,
        logging_steps=settings.logging_steps,
        save_strategy="no",
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
        processing_class=student_tokenizer(), peft_config=lora, callbacks=[LogToFile()],
    )
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    trainer.model.save_pretrained(output)

    evals = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
    return {
        **result.metrics,
        "optimizer_steps": trainer.state.global_step,
        "eval_loss": evals[-1]["eval_loss"] if evals else None,
        "gpu_free_at_start_mib": round(free_at_start / 2**20),
        "peak_gpu_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_gpu_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
        "log_history": trainer.state.log_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of the student on one API's kept rollouts.")
    parser.add_argument("--api", required=True, choices=[*APIS, EVERY_API],
                        help=f"{EVERY_API}: single_trained's adapter, on every API's examples with every tool")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--config", type=Path, default=FACTORY_CONFIG, help="factory config (its train: section)")
    parser.add_argument("--data", type=Path, nargs="+", help="default: data/generated/<api>/round_1..<n>.jsonl")
    parser.add_argument("--output", type=Path, help="adapter folder (default adapters/<api>_r<n>)")
    parser.add_argument("--run-id", help="default: train_<api>_r<n>")
    parser.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (dry run)")
    parser.add_argument("--limit", type=int, help="use only this many examples, a seeded sample (dry run)")
    args = parser.parse_args()

    settings = TrainSettings.model_validate(load_factory_config(args.config)["train"])
    name = f"{args.api}_r{args.round}"
    output = (args.output or REPO / "adapters" / name).resolve()
    run_id = args.run_id or f"train_{name}"
    run_dir = REPO / "runs" / run_id
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} isn't empty; remove it or pick another --output.")
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists; remove it or pick another --run-id.")

    paths = [path.resolve() for path in args.data] if args.data else data_files(args.api, args.round)
    examples = load_examples(paths, args.api)
    if args.limit:
        examples = random.Random(settings.seed).sample(examples, min(args.limit, len(examples)))
    train_examples, eval_examples = split(examples, settings.eval_fraction, settings.seed)
    tokenizer = student_tokenizer()
    every_tool = args.api == EVERY_API
    rows = {example.id: tokenize(with_every_tool(example) if every_tool else example.text, tokenizer)
            for example in examples}
    if too_long := [example_id for example_id, row in rows.items() if len(row["input_ids"]) > settings.max_length]:
        raise SystemExit(f"{len(too_long)} examples are longer than max_length {settings.max_length} "
                         f"(for example {too_long[0]}); raise it in {shown(args.config.resolve())}.")
    train_rows = [rows[example.id] for example in train_examples]
    eval_rows = [rows[example.id] for example in eval_examples]

    run_dir.mkdir(parents=True)
    student = yaml.safe_load(MODELS_CONFIG.read_text())["models"]["student"]
    provenance = {"run_id": run_id, "base_model": student["hf_id"], "revision": student["revision"],
                  "data": [{"path": shown(path), "sha256": sha256(path)} for path in paths],
                  "text": ("re-rendered with every API's tools: " + ", ".join(tool["name"] for tool in tool_schemas(APIS))
                           if every_tool else "stored (each example's own API's tools)"),
                  **git_commit(run_dir)}
    (run_dir / "config.json").write_text(json.dumps({
        **provenance,
        "args": {key: jsonable(value) for key, value in vars(args).items()},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "adapter": shown(output),
        "settings": settings.model_dump(),
        "train": token_counts(train_rows),
        "eval": token_counts(eval_rows),
        "held_out_ids": sorted(example.id for example in eval_examples),
    }, indent=2))
    print(f"{run_id}: {len(train_rows)} training examples, {len(eval_rows)} held out; adapter -> {shown(output)}")

    metrics = train(settings, train_rows, eval_rows, output, run_dir, args.max_steps)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "training_run.json").write_text(json.dumps(provenance, indent=2))

    eval_loss = f"{metrics['eval_loss']:.4f}" if metrics["eval_loss"] is not None else "none"
    print(f"{run_id}: {metrics['optimizer_steps']} optimizer steps in {metrics['train_runtime'] / 60:.1f} min; "
          f"train loss {metrics['train_loss']:.4f}, eval loss {eval_loss}; "
          f"peak GPU memory {metrics['peak_gpu_memory_reserved_mib']:,} MiB reserved")
    print(f"Adapter in {shown(output)}\nRun in {shown(run_dir)}")


if __name__ == "__main__":
    main()
