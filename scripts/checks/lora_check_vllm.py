"""Throwaway LoRA check (task 0.2), part 2: load the adapters built by
lora_make_adapters.py into a running vLLM server, compare greedy outputs with
HF+PEFT, and sanity-check the parsers."""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


def post(url: str, body: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--served-name", default="nano4b")
    parser.add_argument("--adapters", type=Path, required=True)
    args = parser.parse_args()
    hf = json.loads((args.adapters / "hf_results.json").read_text())

    loaded = [args.served_name]
    report: dict = {"load": {}, "compare": {}, "toolcall": {}}
    for name in hf["hf"]:
        if name == "base":
            continue
        status, text = post(f"{args.base_url}/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(args.adapters / name)})
        report["load"][name] = {"status": status, "detail": text[:400]}
        print(f"load {name}: {status} {text[:200]}")
        if status == 200:
            loaded.append(name)

    vllm_ids: dict[str, list[int]] = {}
    for model in loaded:
        status, text = post(f"{args.base_url}/v1/completions", {
            "model": model, "prompt": hf["prompt_ids"], "max_tokens": hf["max_new_tokens"],
            "temperature": 0.0, "return_token_ids": True,
        })
        if status != 200:
            print(f"completion {model}: {status} {text[:300]}")
            continue
        choice = json.loads(text)["choices"][0]
        vllm_ids[model] = choice.get("token_ids") or []
        key = "base" if model == args.served_name else model
        hf_ids = hf["hf"][key]["ids"]
        report["compare"][key] = {
            "vllm_text": choice["text"], "hf_text": hf["hf"][key]["text"],
            "prefix_match_tokens": common_prefix(vllm_ids[model], hf_ids), "of": len(hf_ids),
        }
    base_v = vllm_ids.get(args.served_name, [])
    for key, row in report["compare"].items():
        model = args.served_name if key == "base" else key
        row["same_as_vllm_base"] = vllm_ids.get(model) == base_v
        print(f"{key:15s} vLLM==HF prefix {row['prefix_match_tokens']}/{row['of']}  same as base: {row['same_as_vllm_base']}")
        print(f"   vLLM: {row['vllm_text']!r}\n   HF:   {row['hf_text']!r}")

    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }}]
    for thinking in (False, True):
        status, text = post(f"{args.base_url}/v1/chat/completions", {
            "model": args.served_name, "messages": [{"role": "user", "content": "What's the weather in Raleigh?"}],
            "tools": tools, "temperature": 0.0, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": thinking},
        })
        msg = json.loads(text)["choices"][0]["message"] if status == 200 else {"error": text[:300]}
        row = {"status": status, "tool_calls": [(c["function"]["name"], c["function"]["arguments"]) for c in msg.get("tool_calls") or []],
               "content": (msg.get("content") or "")[:200], "reasoning_chars": len(msg.get("reasoning") or msg.get("reasoning_content") or "")}
        report["toolcall"][f"thinking_{thinking}"] = row
        print(f"toolcall thinking={thinking}: {row}")

    (args.adapters / "vllm_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
