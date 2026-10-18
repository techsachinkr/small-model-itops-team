"""Serving benchmarks (task 4.1) on our real agent traffic, replayed with vLLM's own `vllm bench serve`:
(1) one vLLM server with every specialist adapter against separate servers per merged specialist, GPU memory and
throughput, and (2) 1, 8 and 32 simultaneous agent sessions on each, throughput and p50/p95 latency.

    uv run python -m bench.traffic && uv run python -m bench.merge    # once: the traffic and the merged models
    scripts/stop_vllm.sh student                                       # frees its GPU share; Lightning may stay up
    uv run python -m bench.run [--configuration A B1 B2] [--sessions 1 8 32] [--run-id <id>]
    scripts/serve_student.sh                                           # afterwards

For each configuration this starts its servers one at a time (vLLM profiles the free memory at startup), waits until
they answer, measures the GPU memory they took (nvidia-smi before and after) and reads what vLLM logged about it,
asks each specialist for a few greedy completions, replays the traffic at each number of sessions, then stops the
servers and waits for the memory to come back:
  A   one server: the student with helpdesk_r1 and ops_r1 as LoRA adapters, scripts/serve_student.sh's flags, 0.15
      of the GPU
  B1  two servers, one merged specialist each (bench/merge.py), 0.075 of the GPU each: A's memory in all
  B2  two servers, one merged specialist each, 0.15 of the GPU each: A's memory per server, twice A's in all
Every request is a recorded prompt (bench/traffic.py), sent raw (--skip-chat-template) in session order with
--ignore-eos --custom-output-len -1 for its recorded output length, so every configuration does the same work, and each specialist's
requests go to its own adapter or server. N sessions are `--max-concurrency` N (one request in flight per session:
the tools answer in milliseconds, so a session is nearly always waiting on the model). With 1 session the two
specialists run one after the other on the first --requests-at-1 calls of each; with N > 1, both at once with N/2
sessions each on all their calls. Throughput is both runs' tokens over their combined duration (summed when one
after the other, the longer one when together); latency percentiles are over both runs' requests.

Writes runs/bench/<run_id>/: config.json (arguments, git commit, every server command), per configuration its server
logs and the `vllm bench serve` results (--save-detailed), and results.json, rewritten after each configuration; then
runs/bench/summary.md, a table per measurement with the server flags beside it.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from common.eval_config import load_eval_config
from common.llm import MODELS_CONFIG
from eval.run_eval import REPO, git_commit, stop_servers

BENCH = REPO / "runs" / "bench"
TRAFFIC = BENCH / "traffic"
MERGED = BENCH / "merged"
SUMMARY = BENCH / "summary.md"
VLLM = REPO / ".venv-vllm" / "bin" / "vllm"
APIS = ("helpdesk", "ops")
ADAPTERS = {"helpdesk": "helpdesk_r1", "ops": "ops_r1"}
SESSIONS = (1, 8, 32)
REQUESTS_AT_1 = 40  # per specialist: one session at a time is slow, and the latency there varies little
GREEDY_PROMPTS = 8  # per specialist, compared across configurations
GREEDY_TOKENS = 32
START_TIMEOUT_S = 900
SERVER_MARGIN_MIB = 3072  # what a server takes beyond its share (task 0.2: about 2 GB for the CUDA context and more)
# What this vLLM version logs about memory at startup
LOGGED = {
    "weights_gib": re.compile(r"Model loading took ([\d.]+) GiB"),
    "cache_gib": re.compile(r"Available KV cache memory: ([\d.]+) GiB"),
    "cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "max_concurrency": re.compile(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x"),
}
LORA_FLAGS = ("--enable-lora", "--max-lora-rank", "16", "--max-loras", "4")


class ServerFailed(RuntimeError):
    pass


def student_flags(models: dict[str, Any]) -> tuple[str, ...]:
    """scripts/serve_student.sh's flags, except the model, name, port, memory share, sequence limit and LoRA."""
    student = models["student"]
    return ("--max-model-len", "32768", "--mamba-ssm-cache-dtype", "float32", "--enable-auto-tool-choice",
            "--tool-call-parser", student["tool_call_parser"], "--reasoning-parser", student["reasoning_parser"],
            "--structured-outputs-config", '{"backend": "xgrammar", "disable_any_whitespace": true}')


@dataclass(frozen=True)
class Server:
    label: str  # the model name it serves
    port: int
    share: float  # --gpu-memory-utilization
    command: tuple[str, ...]


@dataclass(frozen=True)
class Configuration:
    name: str
    description: str
    servers: tuple[Server, ...]
    routes: dict[str, tuple[int, str, str | None]]  # API -> its server's port, the served model, the LoRA module


def configurations(models: dict[str, Any]) -> dict[str, Configuration]:
    student, flags = models["student"], student_flags(models)

    def server(model: tuple[str, ...], name: str, port: int, share: float, seqs: int, extra: tuple[str, ...] = ()) -> Server:
        return Server(name, port, share, (str(VLLM), "serve", *model, "--served-model-name", name, "--port", str(port),
                                          *flags, "--gpu-memory-utilization", str(share), "--max-num-seqs", str(seqs), *extra))

    loras = tuple(f"{name}={REPO / 'adapters' / name}" for name in ADAPTERS.values())
    a = server((student["hf_id"], "--revision", student["revision"]), "student", 8011, 0.15, 256,
               (*LORA_FLAGS, "--lora-modules", *loras))

    def merged(name: str, description: str, share: float, seqs: int) -> Configuration:
        servers = {api: server((str(MERGED / adapter),), adapter, port, share, seqs)
                   for (api, adapter), port in zip(ADAPTERS.items(), (8012, 8013))}
        return Configuration(name, description, tuple(servers.values()),
                             {api: (s.port, s.label, None) for api, s in servers.items()})

    return {
        "A": Configuration("A", "one server: the student with both adapters as LoRA", (a,),
                           {api: (a.port, "student", adapter) for api, adapter in ADAPTERS.items()}),
        # 16 sequences per server cover N/2 sessions up to 32; at 0.075 there's little room for more
        "B1": merged("B1", "two servers, one merged specialist each, A's memory in all", 0.075, 16),
        "B2": merged("B2", "two servers, one merged specialist each, A's memory per server", 0.15, 256),
    }


def gpu_memory() -> tuple[int, int]:
    """Used and total MiB on the GPU."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    used, total = (int(part) for part in out.splitlines()[0].split(","))
    return used, total


def memory_refusal(config: Configuration, used: int, total: int) -> str | None:
    need = sum(int(s.share * total) + SERVER_MARGIN_MIB for s in config.servers)
    if total - used < need:
        return (f"{config.name} needs about {need:,} MiB of GPU memory and {total - used:,} are free. Stop the student "
                "server first (scripts/stop_vllm.sh student); Lightning may stay up.")
    return None


def answers(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def vllm_env() -> dict[str, str]:
    """As scripts/vllm_lib.sh: the vLLM venv first on PATH (FlashInfer runs ninja from it), and no Hub access."""
    return os.environ | {"PATH": f"{VLLM.parent}{os.pathsep}{os.environ.get('PATH', '')}", "HF_HUB_OFFLINE": "1"}


def start(server: Server, log_path: Path) -> subprocess.Popen:
    log = log_path.open("w")
    log.write(" ".join(server.command) + "\n")
    log.flush()
    process = subprocess.Popen(server.command, cwd=REPO, env=vllm_env(), stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    deadline = time.monotonic() + START_TIMEOUT_S
    while not answers(server.port):
        if process.poll() is not None:
            raise ServerFailed(f"{server.label} on port {server.port} exited during startup; see {log_path}")
        if time.monotonic() > deadline:
            stop_servers(process)
            raise ServerFailed(f"{server.label} on port {server.port} not ready after {START_TIMEOUT_S} s; see {log_path}")
        time.sleep(2)
    return process


def logged_memory(log_path: Path) -> dict[str, float]:
    text = log_path.read_text(errors="replace")
    found = {key: pattern.findall(text) for key, pattern in LOGGED.items()}
    return {key: float(values[-1].replace(",", "")) for key, values in found.items() if values}


def wait_for_memory(baseline: int, timeout_s: float = 180) -> int:
    """Waits until the GPU is back to `baseline` (within 1 GiB); returns the MiB used then."""
    deadline = time.monotonic() + timeout_s
    used, _ = gpu_memory()
    while used > baseline + 1024 and time.monotonic() < deadline:
        time.sleep(3)
        used, _ = gpu_memory()
    return used


def traffic(api: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (TRAFFIC / f"{api}.jsonl").read_text().splitlines() if line.strip()]


def greedy(port: int, model: str, prompts: list[str]) -> list[str]:
    texts = []
    for prompt in prompts:
        body = {"model": model, "prompt": prompt, "max_tokens": GREEDY_TOKENS, "temperature": 0}
        request = urllib.request.Request(f"http://localhost:{port}/v1/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            texts.append(json.load(response)["choices"][0]["text"])
    return texts


def bench_command(port: int, model: str, lora: str | None, api: str, sessions: int, requests: int, result: Path,
                  tokenizer: Path, sampling: dict[str, Any]) -> list[str]:
    command = [str(VLLM), "bench", "serve", "--backend", "openai", "--endpoint", "/v1/completions",
               "--base-url", f"http://localhost:{port}", "--model", model, "--tokenizer", str(tokenizer),
               "--dataset-name", "custom", "--dataset-path", str(TRAFFIC / f"{api}.jsonl"), "--skip-chat-template",
               # -1: each call's own output_tokens; the default, 256, replaces them (the first run, 20261018-074423_lora)
               "--custom-output-len", "-1",
               "--disable-shuffle", "--no-oversample", "--ignore-eos", "--num-prompts", str(requests),
               "--max-concurrency", str(sessions), "--request-rate", "inf", "--num-warmups", "2", "--seed", "0",
               "--temperature", str(sampling["temperature"]), "--top-p", str(sampling["top_p"]),
               "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95",
               "--save-result", "--save-detailed", "--result-dir", str(result.parent), "--result-filename", result.name,
               "--disable-tqdm"]
    return command + (["--lora-modules", lora] if lora else [])


def percentiles(values: list[float], scale: float) -> dict[str, float] | None:
    if len(values) < 2:
        return None
    q = statistics.quantiles(values, n=100, method="inclusive")
    return {"p50": q[49] * scale, "p95": q[94] * scale}


def combine(runs: list[dict[str, Any]], one_after_the_other: bool) -> dict[str, Any]:
    """Both specialists' `vllm bench serve` results as one measurement. A run without results counts as failed."""
    missing = [run["api"] for run in runs if "duration" not in run]
    done = [run for run in runs if "duration" in run]
    ttfts, tpots, e2els = [], [], []
    for run in done:
        for ttft, itl, error in zip(run["ttfts"], run["itls"], run["errors"]):
            if error:
                continue
            ttfts.append(ttft)
            e2els.append(ttft + sum(itl))
            if itl:
                tpots.append(sum(itl) / len(itl))
    durations = [run["duration"] for run in done] or [0.0]
    duration = sum(durations) if one_after_the_other else max(durations)
    completed = sum(run["completed"] for run in done)
    output_tokens = sum(run["total_output_tokens"] for run in done)
    input_tokens = sum(run["total_input_tokens"] for run in done)
    return {
        "completed": completed,
        "failed": sum(run.get("failed", 0) for run in done),
        "missing_runs": missing,
        "duration_s": duration,
        "requests_per_s": completed / duration if duration else 0.0,
        "output_tokens_per_request": output_tokens / completed if completed else 0.0,  # as generated
        "output_tokens_per_s": output_tokens / duration if duration else 0.0,
        "total_tokens_per_s": (input_tokens + output_tokens) / duration if duration else 0.0,
        "ttft_ms": percentiles(ttfts, 1000),
        "tpot_ms": percentiles(tpots, 1000),
        "e2el_s": percentiles(e2els, 1),
    }


def replay(config: Configuration, sessions: int, folder: Path, tokenizer: Path, sampling: dict[str, Any],
           requests_at_1: int) -> dict[str, Any]:
    calls = {api: len(traffic(api)) for api in APIS}
    groups = ([[(api, 1, min(requests_at_1, calls[api]))] for api in APIS] if sessions == 1
              else [[(api, sessions // 2, calls[api]) for api in APIS]])
    runs = []
    for group in groups:
        started = []
        for api, concurrency, requests in group:
            port, model, lora = config.routes[api]
            result = folder / f"s{sessions}_{api}.json"
            log = (folder / f"s{sessions}_{api}.log").open("w")
            command = bench_command(port, model, lora, api, concurrency, requests, result, tokenizer, sampling)
            log.write(" ".join(command) + "\n")
            log.flush()
            started.append((api, result, subprocess.Popen(command, cwd=REPO, env=vllm_env(), stdout=log,
                                                          stderr=subprocess.STDOUT)))
        for api, result, process in started:
            process.wait()
            runs.append({"api": api, "returncode": process.returncode,
                         **(json.loads(result.read_text()) if result.exists() else {})})
    return combine(runs, one_after_the_other=sessions == 1)


def run_configuration(config: Configuration, folder: Path, args: argparse.Namespace, tokenizer: Path,
                      sampling: dict[str, Any]) -> dict[str, Any]:
    folder.mkdir(parents=True, exist_ok=True)
    baseline, total = gpu_memory()
    entry: dict[str, Any] = {"description": config.description,
                             "servers": [{"label": s.label, "share": s.share, "command": " ".join(s.command)}
                                         for s in config.servers],
                             "measurements": {}}
    if reason := memory_refusal(config, baseline, total):
        raise SystemExit(f"Refusing to start {config.name}: {reason}")
    processes = []
    try:
        for server in config.servers:
            processes.append(start(server, folder / f"{server.label}.log"))
        time.sleep(5)
        entry["gpu_memory_mib"] = gpu_memory()[0] - baseline
        for server, record in zip(config.servers, entry["servers"]):
            record.update(logged_memory(folder / f"{server.label}.log"))
        entry["greedy"] = {api: greedy(port, lora or model, [c["prompt"] for c in traffic(api)[:GREEDY_PROMPTS]])
                           for api, (port, model, lora) in config.routes.items()}
        for sessions in args.sessions:
            measurement = replay(config, sessions, folder, tokenizer, sampling, args.requests_at_1)
            entry["measurements"][str(sessions)] = measurement
            print(f"  {config.name}, {sessions} sessions: {measurement['completed']} requests "
                  f"({measurement['failed']} failed), {measurement['output_tokens_per_s']:.0f} output tok/s")
    except ServerFailed as error:
        entry["error"] = str(error)
        print(f"  {config.name}: {error}")
    finally:
        for process in processes:
            stop_servers(process)
        entry["gpu_memory_after_mib"] = wait_for_memory(baseline)
    return entry


def fmt(value: dict[str, float] | None, digits: int) -> str:
    return f"{value['p50']:.{digits}f} / {value['p95']:.{digits}f}" if value else "-"


def summary(results: dict[str, Any], traffic_stats: dict[str, Any]) -> str:
    configs = results["configurations"]
    lines = [
        "# Serving benchmarks (task 4.1)",
        "",
        f"Run `{results['run_id']}` ({results['started_at'][:16]} UTC, commit `{results['git_commit'][:7]}`"
        f"{', with uncommitted changes' if results['git_dirty'] else ''}), built by `bench/run.py`. One H200; "
        "Lightning's server was up and idle throughout. Traffic: every model call of the r1 practice runs "
        "(`bench/traffic.py`), replayed with `vllm bench serve` as recorded:",
        "",
        "| specialist | run | sessions | calls (tool calls, answers, self-checks) | prompt tokens | output tokens |",
        "|---|---|---|---|---|---|",
    ]
    for api, stats in traffic_stats.get("apis", {}).items():
        calls = stats["calls"]
        lines.append(f"| {api} | `{stats['run']}` | {stats['sessions']} | {sum(calls.values())} ({calls.get('tool_call', 0)}, "
                     f"{calls.get('answer', 0)}, {calls.get('self_check', 0)}) | {stats['prompt_tokens']:,} "
                     f"(run: {stats['recorded_prompt_tokens']:,}) | {stats['output_tokens']:,} "
                     f"(run: {stats['recorded_completion_tokens']:,}) |")
    lines += ["", "## 1. GPU memory: one server with both adapters against one merged server per specialist", "",
              "| configuration | servers | flags | GPU memory taken | weights | KV and Mamba cache | cache tokens | "
              "max concurrency at 32k tokens |", "|---|---|---|---|---|---|---|---|"]
    for name, entry in configs.items():
        servers = entry["servers"]

        def each(key: str, unit: str = "", digits: int = 2) -> str:
            values = [s.get(key) for s in servers]
            return " + ".join("-" if v is None else f"{v:,.{digits}f}{unit}" for v in values)

        flags = " + ".join(f"{s['share']} of the GPU" for s in servers) + (", LoRA" if "--enable-lora" in servers[0]["command"] else ", merged")
        taken = f"{entry['gpu_memory_mib']:,} MiB" if "gpu_memory_mib" in entry else "-"
        lines.append(f"| {name}: {entry['description']} | {len(servers)} | {flags} | {taken} | {each('weights_gib', ' GiB')} | "
                     f"{each('cache_gib', ' GiB')} | {each('cache_tokens', '', 0)} | {each('max_concurrency', 'x')} |")
    lines += ["", "## 1 and 2. Throughput and latency at each number of simultaneous sessions", "",
              "| configuration | sessions | requests (failed) | output tokens per request | output tok/s | total tok/s | "
              "requests/s | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms) | end to end p50 / p95 (s) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, entry in configs.items():
        if "error" in entry and not entry["measurements"]:
            lines.append(f"| {name} | - | {entry['error']} | | | | | | | |")
        for sessions, m in entry["measurements"].items():
            partial = f" {m['missing_runs']} missing" if m["missing_runs"] else ""
            per_request = m.get("output_tokens_per_request")
            if per_request is None:  # results from before the column existed
                per_request = m["output_tokens_per_s"] * m["duration_s"] / m["completed"] if m["completed"] else 0.0
            lines.append(f"| {name} | {sessions} | {m['completed']} ({m['failed']}){partial} | {per_request:.1f} | "
                         f"{m['output_tokens_per_s']:,.0f} | {m['total_tokens_per_s']:,.0f} | {m['requests_per_s']:.2f} | "
                         f"{fmt(m['ttft_ms'], 0)} | {fmt(m['tpot_ms'], 1)} | {fmt(m['e2el_s'], 2)} |")
    reference = configs.get("A", {}).get("greedy")
    if reference:
        lines += ["", f"## The merged models against the adapters", "",
                  f"Greedy completions ({GREEDY_TOKENS} tokens) of the first {GREEDY_PROMPTS} calls of each specialist, "
                  "identical to configuration A's (BF16 merges can round differently):", "",
                  "| configuration | identical to A |", "|---|---|"]
        for name, entry in configs.items():
            if name != "A" and entry.get("greedy"):
                pairs = [(a, b) for api in APIS for a, b in zip(reference[api], entry["greedy"][api])]
                lines.append(f"| {name} | {sum(a == b for a, b in pairs)}/{len(pairs)} |")
    lines += ["", "## Server flags", ""]
    for name, entry in configs.items():
        for server in entry["servers"]:
            lines += [f"{name}, {server['label']}:", "", f"    {server['command'].replace(str(REPO) + '/', '')}", ""]
    if results.get("output_len") == "recorded":
        lengths = ("with `--ignore-eos --custom-output-len -1` for each call's recorded output length (the table's "
                   "output tokens per request are what the servers generated)")
    else:
        lengths = ("with `--ignore-eos` and vLLM's default `--custom-output-len` of 256: every request generated 256 "
                   "tokens, not its recorded length (about 66 on average). This run predates the fix in bench/run.py")
    lines += ["## Method", "",
              "- Requests are the recorded prompts sent raw (`--skip-chat-template`, in session order with "
              f"`--disable-shuffle`) {lengths}; sampling as `configs/eval.yaml`. Self-checks were structured outputs in "
              "the runs; here they are plain completions.",
              f"- N sessions are `--max-concurrency` N. With 1 session the two specialists run one after the other on "
              f"their first {results['requests_at_1']} calls each; with more, both at once with N/2 sessions each on "
              "all their calls, each to its own adapter (A) or server (B). Throughput: both runs' tokens over the "
              "summed (one after the other) or longer (together) duration. Two warm-up requests per run.",
              "- GPU memory taken: nvidia-smi's used memory with the configuration's servers up, minus before they "
              "started. Weights, cache and max concurrency: the servers' own startup log."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serving benchmarks on the specialists' recorded traffic.")
    parser.add_argument("--configuration", nargs="+", default=["A", "B1", "B2"], choices=["A", "B1", "B2"])
    parser.add_argument("--sessions", nargs="+", type=int, default=list(SESSIONS))
    parser.add_argument("--requests-at-1", type=int, default=REQUESTS_AT_1)
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "eval.yaml", help="eval config (sampling)")
    parser.add_argument("--run-id", help="default: <time>_lora")
    args = parser.parse_args()
    if any(n > 1 and n % 2 for n in args.sessions):
        parser.error("more than 1 session must be even: each specialist gets half")

    models = yaml.safe_load(MODELS_CONFIG.read_text())["models"]
    selected = [configurations(models)[name] for name in args.configuration]
    missing = [str(TRAFFIC / f"{api}.jsonl") for api in APIS if not (TRAFFIC / f"{api}.jsonl").exists()]
    missing += [str(MERGED / adapter) for adapter in ADAPTERS.values() if any(c.name != "A" for c in selected)
                and not (MERGED / adapter / "model.safetensors").exists()]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)}: run bench.traffic and bench.merge first.")
    busy = sorted({s.port for c in selected for s in c.servers if answers(s.port)})
    if busy:
        raise SystemExit(f"Something already answers on port(s) {busy}.")

    from bench.merge import student_snapshot

    _, tokenizer = student_snapshot()
    sampling = load_eval_config(args.config).sampling()
    started = datetime.now(timezone.utc)
    run_id = args.run_id or f"{started:%Y%m%d-%H%M%S}_lora"
    run_dir = BENCH / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    traffic_stats = json.loads((TRAFFIC / "traffic.json").read_text())
    results: dict[str, Any] = {"run_id": run_id, "started_at": started.isoformat(), "requests_at_1": args.requests_at_1,
                               "sessions": args.sessions, "sampling": sampling, "output_len": "recorded",
                               **git_commit(run_dir), "configurations": {}}
    (run_dir / "config.json").write_text(json.dumps({
        **{k: v for k, v in results.items() if k != "configurations"}, "traffic": traffic_stats,
        "servers": {c.name: [" ".join(s.command) for s in c.servers] for c in selected},
    }, indent=2))
    try:
        for config in selected:
            print(f"{config.name}: {config.description}")
            results["configurations"][config.name] = run_configuration(config, run_dir / config.name, args, tokenizer,
                                                                       sampling)
            (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    finally:  # what finished, also when a configuration was refused or the run was stopped
        if results["configurations"]:
            from bench.structured import SECTION_END, SECTION_START, with_section

            text = summary(results, traffic_stats)
            old = SUMMARY.read_text() if SUMMARY.exists() else ""
            if SECTION_START in old:  # keep bench/structured.py's section
                text = with_section(text, old[old.index(SECTION_START):old.index(SECTION_END) + len(SECTION_END)] + "\n")
            SUMMARY.write_text(text)
            print(f"Results in {run_dir}; summary in {SUMMARY}")


if __name__ == "__main__":
    main()
