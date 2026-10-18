"""Merges each specialist's LoRA adapter into a full copy of the student, for the separate-servers configurations of
the serving benchmarks (task 4.1, measurement 1).

    uv run python -m bench.merge [--adapter helpdesk_r1 --adapter ops_r1] [--output-dir runs/bench/merged]

The merge works on the checkpoint's own tensors, so vLLM loads the result exactly as it loads the student: the
student's model.safetensors at its pinned revision (configs/models.yaml), with W + (lora_alpha / r) * B @ A added to
every weight the adapter trained (computed in float32, stored in BF16, the checkpoint's dtype), and every other file
of the snapshot (config, tokenizer, chat template) copied unchanged. Saving through transformers would rename the
tensors: transformers 5 loads the checkpoint's `backbone.layers.N...` as `model.layers.N...`, which is also how the
adapter names them (`base_model.model.model.layers.N...lora_A.weight`). Writes <output-dir>/<adapter>/ (about 8 GB)
with merge.json (base, revision, adapter, scaling, the weights merged, git commit). CPU only: the GPU's servers are
untouched.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

from common.llm import MODELS_CONFIG
from eval.run_eval import REPO, git

os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))  # where the student is cached

OUTPUT_DIR = REPO / "runs" / "bench" / "merged"
ADAPTERS = ("helpdesk_r1", "ops_r1")
PEFT_PREFIX = "base_model.model.model."  # PEFT's wrapper, then transformers' name for the checkpoint's backbone
CHECKPOINT_PREFIX = "backbone."


def checkpoint_key(adapter_key: str) -> tuple[str, str]:
    """The checkpoint weight an adapter tensor belongs to, and which half of the pair it is (lora_A or lora_B)."""
    if not adapter_key.startswith(PEFT_PREFIX) or not adapter_key.endswith((".lora_A.weight", ".lora_B.weight")):
        raise ValueError(f"unexpected adapter tensor {adapter_key!r}")
    module, part, _ = adapter_key[len(PEFT_PREFIX):].rsplit(".", 2)
    return f"{CHECKPOINT_PREFIX}{module}.weight", part


def merge_weights(base: dict[str, torch.Tensor], adapter: dict[str, torch.Tensor], scaling: float) -> list[str]:
    """Adds every LoRA pair's update to its base weight, in place. Returns the weights merged."""
    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for key, tensor in adapter.items():
        target, part = checkpoint_key(key)
        pairs[target][part] = tensor
    for target, pair in sorted(pairs.items()):
        if target not in base or set(pair) != {"lora_A", "lora_B"}:
            raise ValueError(f"{target}: no such checkpoint weight, or not a complete lora_A/lora_B pair")
        delta = scaling * (pair["lora_B"].float() @ pair["lora_A"].float())
        if delta.shape != base[target].shape:
            raise ValueError(f"{target}: update {tuple(delta.shape)} doesn't fit the weight {tuple(base[target].shape)}")
        base[target] = (base[target].float() + delta).to(base[target].dtype)
    return sorted(pairs)


def scaling_of(adapter_config: dict[str, Any]) -> float:
    if adapter_config.get("use_dora") or adapter_config.get("use_rslora"):
        raise ValueError("only plain LoRA adapters are merged here")
    return adapter_config["lora_alpha"] / adapter_config["r"]


def student_snapshot() -> tuple[dict[str, Any], Path]:
    from huggingface_hub import snapshot_download

    student = yaml.safe_load(MODELS_CONFIG.read_text())["models"]["student"]
    return student, Path(snapshot_download(student["hf_id"], revision=student["revision"], local_files_only=True))


def merge(name: str, output_dir: Path) -> Path:
    student, snapshot = student_snapshot()
    adapter_dir = REPO / "adapters" / name
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    (weights_file,) = snapshot.glob("*.safetensors")  # the student has one
    base = load_file(weights_file)
    merged = merge_weights(base, load_file(adapter_dir / "adapter_model.safetensors"), scaling_of(adapter_config))
    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    save_file(base, out / weights_file.name, metadata={"format": "pt"})
    for path in snapshot.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copy(path, out / path.name)  # follows the cache's symlinks
    (out / "merge.json").write_text(json.dumps({
        "base": student["hf_id"], "revision": student["revision"], "adapter": str(adapter_dir.relative_to(REPO)),
        "scaling": scaling_of(adapter_config), "weights_merged": len(merged), "git_commit": git("rev-parse", "HEAD").strip(),
    }, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into full copies of the student.")
    parser.add_argument("--adapter", action="append", help=f"an adapter in adapters/; default {' and '.join(ADAPTERS)}")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    for name in args.adapter or ADAPTERS:
        out = merge(name, args.output_dir)
        print(f"{name}: {json.loads((out / 'merge.json').read_text())['weights_merged']} weights merged into {out}")


if __name__ == "__main__":
    main()
