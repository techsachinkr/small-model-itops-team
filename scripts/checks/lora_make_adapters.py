"""Throwaway LoRA check (task 0.2): build untrained PEFT adapters for the student
base and record HF greedy outputs, to compare with vLLM serving the same adapters."""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ATTN_MLP = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
MAMBA = ["in_proj"]  # PEFT rejects out_proj and conv1d on nemotron_h
# name -> (target modules, random B). Zero B (PEFT default) must reproduce the base model.
SPECS: dict[str, tuple[list[str], bool]] = {
    "attn_mlp_zero": (ATTN_MLP, False),
    "attn_mlp_rand": (ATTN_MLP, True),
    "mamba_rand": (MAMBA, True),
}
REL_NORM = 0.05  # random adapters perturb each weight by ~5% (Frobenius), so text stays readable
PROMPT = "The payments deployment in prod keeps crashing after this morning's deploy. What should I check first? Answer in two sentences."
MAX_NEW = 32


def randomize_b(model: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for module in model.modules():
        if not hasattr(module, "lora_B") or "default" not in module.lora_B:
            continue
        a = module.lora_A["default"].weight.float()
        b = torch.randn(module.lora_B["default"].weight.shape, generator=gen).to(a.device)
        delta = module.scaling["default"] * (b @ a)
        target = REL_NORM * module.base_layer.weight.float().norm()
        module.lora_B["default"].weight.data = (b * (target / delta.norm())).to(module.lora_B["default"].weight.dtype)


def greedy(model, input_ids: torch.Tensor) -> list[int]:
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=MAX_NEW, do_sample=False)
    return out[0, input_ids.shape[1]:].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False,
    )
    input_ids = torch.tensor([prompt_ids], device="cuda")

    results = {"prompt_ids": prompt_ids, "max_new_tokens": MAX_NEW, "hf": {}}
    base_ids = greedy(model, input_ids)
    results["hf"]["base"] = {"ids": base_ids, "text": tok.decode(base_ids)}
    print("base:", repr(tok.decode(base_ids)), flush=True)

    for i, (name, (targets, rand_b)) in enumerate(SPECS.items()):
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=targets, task_type="CAUSAL_LM")
        peft_model = get_peft_model(model, cfg)
        if rand_b:
            randomize_b(peft_model, seed=1000 + i)
        ids = greedy(peft_model, input_ids)
        results["hf"][name] = {"ids": ids, "text": tok.decode(ids), "targets": targets}
        print(f"{name}:", repr(tok.decode(ids)), flush=True)
        peft_model.save_pretrained(args.out / name)
        model = peft_model.unload()

    (args.out / "hf_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
