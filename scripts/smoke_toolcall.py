"""Checks that a served model returns parsed tool calls for 20 prompts.

    uv run python scripts/smoke_toolcall.py --model student
    uv run python scripts/smoke_toolcall.py --model ops_r1 --min-parsed 19

Each prompt asks for the weather in a named city, with one dummy get_weather tool. A prompt counts
as parsed if vLLM's tool parser returned a get_weather call whose arguments are a JSON object with a
non-empty "city". Writes runs/smoke_toolcall/<model>_<LLM_MODE>_<time>/ and exits 1 below the pass
mark. Honors LLM_MODE like every call through common/llm.py.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common import llm  # noqa: E402

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, for example Raleigh"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

PROMPTS = [
    "What's the weather in Raleigh right now?",
    "How hot is it in Phoenix today?",
    "Is it raining in Seattle?",
    "Check the current weather in Tokyo for me.",
    "I'm flying to London this afternoon. What's the weather like there?",
    "Should I bring an umbrella in Mumbai today?",
    "Give me the temperature in Toronto in celsius.",
    "What's it like outside in Denver at the moment?",
    "weather berlin",
    "Can you look up the current conditions in Sydney, in fahrenheit?",
    "Is it windy in Chicago today?",
    "Do I need a coat in Oslo right now?",
    "Tell me the weather for São Paulo.",
    "What's the forecast looking like in Nairobi at the moment?",
    "My team in Bangalore says it's pouring. Can you check the weather there?",
    "Is it sunny in Madrid?",
    "Current temperature in Anchorage, please.",
    "How's the weather in Austin, Texas?",
    "Look up the weather in Paris and tell me if it's good for a walk.",
    "Quick check: weather in Singapore?",
]


def parsed_city(message: dict[str, Any]) -> str | None:
    """The city from the first get_weather call, if vLLM parsed one with valid arguments."""
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        if function.get("name") != "get_weather":
            continue
        try:
            args = json.loads(function.get("arguments") or "")
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict) and isinstance(args.get("city"), str) and args["city"].strip():
            return args["city"]
    return None


def git_commit() -> str:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return commit + ("-dirty" if dirty else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="student", help='"lightning", "student" or an adapter name')
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "smoke_toolcall.yaml")
    parser.add_argument("--min-parsed", type=int, help="override the config's pass mark")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if args.min_parsed is not None:
        config["min_parsed"] = args.min_parsed
    sampling = config["sampling"].get(args.model, config["sampling"]["student"])
    mode = os.environ.get("LLM_MODE", "live")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = REPO / "runs" / "smoke_toolcall" / f"{args.model}_{mode}_{stamp}"
    run_dir.mkdir(parents=True)
    resolved = {"model": args.model, "llm_mode": mode, "config_path": str(args.config), **config,
                "sampling_used": sampling, "git_commit": git_commit()}
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2))

    parsed = 0
    with open(run_dir / "results.jsonl", "w") as out:
        for i, prompt in enumerate(PROMPTS):
            response = llm.chat(
                args.model, [{"role": "user", "content": prompt}], tools=[WEATHER_TOOL],
                max_tokens=config["max_tokens"], seed=config["seed"] + i, **sampling,
            )
            city = parsed_city(response.message)
            parsed += city is not None
            out.write(json.dumps({"i": i, "prompt": prompt, "city": city, **response.model_dump()}) + "\n")
            shown = f"get_weather(city={city!r})" if city else f"NO CALL: {str(response.message.get('content'))[:80]!r}"
            print(f"{i + 1:2d} {'ok  ' if city else 'MISS'} {shown}  "
                  f"[{response.prompt_tokens}+{response.completion_tokens} tok, {response.latency_s:.2f} s]")

    summary = {"model": args.model, "llm_mode": mode, "parsed": f"{parsed}/{len(PROMPTS)}",
               "min_parsed": config["min_parsed"], "passed": parsed >= config["min_parsed"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.model} ({mode}): {parsed}/{len(PROMPTS)} parsed tool calls, "
          f"pass mark {config['min_parsed']}: {'PASS' if summary['passed'] else 'FAIL'}  ({run_dir.relative_to(REPO)})")
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
