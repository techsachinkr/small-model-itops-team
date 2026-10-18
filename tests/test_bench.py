"""bench/ without servers or a GPU: the rebuilt traffic has the runs' own prompt tokens, the merge matches PEFT's, the
benchmark servers keep scripts/serve_student.sh's flags, and measurements and the summary are put together right."""

import json

import pytest
import torch
import yaml

from bench import run as bench_run
from bench.merge import checkpoint_key, merge_weights, scaling_of, student_snapshot
from common.llm import MODELS_CONFIG
from common.seatbelt import SELF_CHECK_PROMPT
from eval.run_eval import REPO

MODELS = yaml.safe_load(MODELS_CONFIG.read_text())["models"]
CONFIGS = bench_run.configurations(MODELS)


def test_the_rebuilt_traffic_has_the_runs_own_prompt_tokens() -> None:
    """Each session's rebuilt prompts add up to exactly the prompt tokens vLLM reported in the run (task 2.2's r1 run),
    and every answer is followed by its self-check, which has no tools."""
    rollouts = REPO / "runs" / "r1_helpdesk" / "rollouts.jsonl"
    if not rollouts.exists():
        pytest.skip("needs runs/r1_helpdesk (task 2.2)")
    traffic = pytest.importorskip("bench.traffic")
    try:
        traffic.student_tokenizer()
    except OSError:
        pytest.skip("needs the student's tokenizer in the Hugging Face cache")
    for line in rollouts.read_text().splitlines()[:4]:
        row = json.loads(line)
        calls = traffic.session_calls(row, "session")
        kinds = [c["kind"] for c in calls]
        assert kinds.count("answer") == kinds.count("self_check") >= 1
        assert all(kinds[i + 1] == "self_check" for i, kind in enumerate(kinds) if kind == "answer")
        assert sum(traffic.tokens(c["prompt"]) for c in calls) == row["trace"]["prompt_tokens"]
        description = row["responses_create_params"]["tools"][0]["description"]
        for c in calls:
            assert (description in c["prompt"]) is (c["kind"] != "self_check") and c["output_tokens"] > 0
        assert SELF_CHECK_PROMPT in calls[kinds.index("self_check")]["prompt"]


def test_the_merge_matches_peft_s_merge_and_unload() -> None:
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)

    class Tiny(torch.nn.Module):  # a Mamba layer's in_proj under the student's module path, as transformers names it
        def __init__(self) -> None:
            super().__init__()
            mixer, layer = torch.nn.Module(), torch.nn.Module()
            mixer.in_proj = torch.nn.Linear(6, 4, bias=False)
            layer.mixer = mixer
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([layer])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model.layers[0].mixer.in_proj(x)

    tiny = Tiny()
    key = "backbone.layers.0.mixer.in_proj.weight"
    base = {key: tiny.model.layers[0].mixer.in_proj.weight.detach().clone()}
    config = peft.LoraConfig(r=2, lora_alpha=4, target_modules=["in_proj"])
    model = peft.get_peft_model(tiny, config)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param)  # PEFT starts B at zero, which would merge nothing
    adapter = peft.get_peft_model_state_dict(model)
    assert merge_weights(base, adapter, scaling_of(config.to_dict())) == [key]
    assert torch.allclose(base[key], model.merge_and_unload().model.layers[0].mixer.in_proj.weight, atol=1e-6)


def test_every_adapter_weight_maps_to_a_checkpoint_weight_it_fits() -> None:
    safetensors = pytest.importorskip("safetensors")
    adapter = REPO / "adapters" / "helpdesk_r1" / "adapter_model.safetensors"
    if not adapter.exists():
        pytest.skip("needs adapters/helpdesk_r1 (task 2.2)")
    try:
        _, snapshot = student_snapshot()
    except Exception:
        pytest.skip("needs the student in the Hugging Face cache")
    (weights,) = snapshot.glob("*.safetensors")
    with safetensors.safe_open(weights, "pt") as f:
        shapes = {key: f.get_slice(key).get_shape() for key in f.keys()}
    with safetensors.safe_open(adapter, "pt") as f:
        keys = list(f.keys())
        for key in keys:
            target, part = checkpoint_key(key)
            rows, columns = shapes[target]
            shape = f.get_slice(key).get_shape()
            assert (shape[1] == columns) if part == "lora_A" else (shape[0] == rows), key
    assert len(keys) == 142  # 71 weights, 7 kinds of projection (task 2.2)


def test_the_benchmark_servers_keep_serve_student_sh_s_flags() -> None:
    script = (REPO / "scripts" / "serve_student.sh").read_text()
    (a,) = CONFIGS["A"].servers
    for flag in ("--max-model-len 32768", "--mamba-ssm-cache-dtype float32", "--enable-auto-tool-choice",
                 "--gpu-memory-utilization 0.15", "--max-num-seqs 256", "--enable-lora --max-lora-rank 16 --max-loras 4",
                 "--structured-outputs-config '{\"backend\": \"xgrammar\", \"disable_any_whitespace\": true}'"):
        assert flag in script
        assert flag.replace("'", "") in " ".join(a.command)
    flags = bench_run.student_flags(MODELS)
    assert MODELS["student"]["tool_call_parser"] in flags and MODELS["student"]["reasoning_parser"] in flags
    for name, share in (("B1", "0.075"), ("B2", "0.15")):
        servers = CONFIGS[name].servers
        assert [s.label for s in servers] == ["helpdesk_r1", "ops_r1"] and "--enable-lora" not in " ".join(servers[0].command)
        assert all(" ".join(flags) in " ".join(s.command) and f"--gpu-memory-utilization {share}" in " ".join(s.command)
                   for s in servers)
    # Each specialist's requests go to its own adapter (A) or its own merged server (B).
    assert CONFIGS["A"].routes == {"helpdesk": (8011, "student", "helpdesk_r1"), "ops": (8011, "student", "ops_r1")}
    assert CONFIGS["B1"].routes == {"helpdesk": (8012, "helpdesk_r1", None), "ops": (8013, "ops_r1", None)}


def test_the_bench_replays_each_call_as_recorded() -> None:
    """vllm bench serve replaces the dataset's output lengths with --custom-output-len, 256 by default, unless it is -1
    (the first run, 20261018-074423_lora, generated 256 tokens per request)."""
    command = " ".join(bench_run.bench_command(8011, "student", "helpdesk_r1", "helpdesk", 4, 147, REPO / "r.json",
                                               REPO / "tokenizer", {"temperature": 0.6, "top_p": 0.95}))
    for flag in ("--custom-output-len -1", "--ignore-eos", "--skip-chat-template", "--disable-shuffle", "--no-oversample",
                 "--lora-modules helpdesk_r1", "--max-concurrency 4", "--num-prompts 147", "--temperature 0.6"):
        assert flag in command


def test_a_configuration_that_doesnt_fit_is_refused() -> None:
    total = 143_771
    assert "stop_vllm.sh student" in bench_run.memory_refusal(CONFIGS["B2"], used=105_067, total=total)  # both up
    assert bench_run.memory_refusal(CONFIGS["B2"], used=81_616, total=total) is None  # Lightning only
    assert bench_run.memory_refusal(CONFIGS["A"], used=81_616, total=total) is None


RUN_A = {"api": "helpdesk", "duration": 10.0, "completed": 2, "failed": 1, "total_input_tokens": 1000,
         "total_output_tokens": 100, "ttfts": [0.1, 0.3, 0.0], "itls": [[0.01, 0.01], [0.02, 0.02], []],
         "errors": ["", "", "timeout"]}
RUN_B = {"api": "ops", "duration": 20.0, "completed": 2, "failed": 0, "total_input_tokens": 3000,
         "total_output_tokens": 300, "ttfts": [0.2, 0.4], "itls": [[0.03], [0.04]], "errors": ["", ""]}


def test_both_specialists_runs_make_one_measurement() -> None:
    together = bench_run.combine([RUN_A, RUN_B], one_after_the_other=False)
    assert (together["duration_s"], together["completed"], together["failed"]) == (20.0, 4, 1)
    assert together["output_tokens_per_s"] == 400 / 20 and together["total_tokens_per_s"] == 4400 / 20
    assert together["output_tokens_per_request"] == 100
    assert together["ttft_ms"]["p50"] == pytest.approx(250)  # 100, 200, 300, 400 ms; the failed request is left out
    assert together["e2el_s"]["p50"] == pytest.approx((0.23 + 0.34) / 2)
    assert bench_run.combine([RUN_A, RUN_B], one_after_the_other=True)["duration_s"] == 30.0
    assert bench_run.combine([RUN_A, {"api": "ops", "returncode": 1}], False)["missing_runs"] == ["ops"]


def test_the_summary_has_a_table_per_measurement_with_the_flags() -> None:
    measurement = bench_run.combine([RUN_A, RUN_B], one_after_the_other=False)
    servers = [{"label": "student", "share": 0.15, "command": f"{REPO}/.venv-vllm/bin/vllm serve student --enable-lora",
                "weights_gib": 7.71, "cache_gib": 11.36, "cache_tokens": 624230.0, "max_concurrency": 19.05}]
    results = {"run_id": "run", "started_at": "2026-10-18T08:00:00+00:00", "git_commit": "abc1234def", "git_dirty": False,
               "requests_at_1": 40, "configurations": {
                   "A": {"description": "one server", "servers": servers, "gpu_memory_mib": 23436,
                         "greedy": {"helpdesk": ["a"], "ops": ["b"]}, "measurements": {"8": measurement}},
                   "B1": {"description": "two servers", "servers": servers * 2, "gpu_memory_mib": 30000,
                          "greedy": {"helpdesk": ["a"], "ops": ["c"]}, "measurements": {"8": measurement}}}}
    text = bench_run.summary(results, {"apis": {}})
    assert "## 1. GPU memory" in text and "## 1 and 2. Throughput and latency" in text and "## Server flags" in text
    assert "| B1 | 1/2 |" in text and "    .venv-vllm/bin/vllm serve student --enable-lora" in text
    assert "| A | 8 | 4 (1) | 100.0 | 20 | 220 | 0.20 | 250 / 385 | 25.0 / 38.5 | 0.28 / 0.42 |" in text
    assert "every request generated 256 tokens" in text  # results without output_len: the first run's default
    recorded = bench_run.summary(results | {"output_len": "recorded"}, {"apis": {}})
    assert "--custom-output-len -1" in recorded and "every request generated 256" not in recorded
    assert "7.71 GiB + 7.71 GiB" in text and "23,436 MiB" in text


def test_structured_answers_are_graded_as_our_code_parses_them() -> None:
    from bench.structured import grade

    schema = {"type": "object", "properties": {"reason": {"type": "string"}, "tool_call_needed": {"type": "boolean"}},
              "required": ["reason", "tool_call_needed"], "additionalProperties": False}
    assert grade(' {"reason": "done", "tool_call_needed": false}\n', schema) == {
        "json": True, "fenced": False, "fits": True, "fits_if_unfenced": False}
    assert grade('{"reason": "done"}', schema)["json"] and not grade('{"reason": "done"}', schema)["fits"]
    fenced = grade('Sure:\n```json\n{"reason": "done", "tool_call_needed": false}\n```', schema)
    assert fenced == {"json": False, "fenced": True, "fits": False, "fits_if_unfenced": True}
    assert grade("No tool call is needed.", schema) == {"json": False, "fenced": False, "fits": False, "fits_if_unfenced": False}


def test_the_structured_requests_are_the_ones_the_evals_send() -> None:
    from agents.orchestrator import plan_schema
    from bench.structured import jobs, plan_requests, self_check_requests, with_section, SECTION_END, SECTION_START

    plans = plan_requests(repeats=3)
    assert len(plans) == 3 * 8 and {p["case"] for p in plans} == {f"pr-multi-0{n}" for n in range(1, 9)}
    assert plans[0]["schema"] == plan_schema(["helpdesk", "ops"])
    assert '"name": "ops"' in plans[0]["messages"][0]["content"] and plans[0]["messages"][1]["role"] == "user"
    if (REPO / "runs" / "r1_helpdesk" / "rollouts.jsonl").exists():
        checks = self_check_requests("r1_helpdesk")
        assert len(checks) == 45  # one per answer (bench/traffic.py: 45 self-checks)
        assert all(c["messages"][-1] == {"role": "user", "content": SELF_CHECK_PROMPT} for c in checks)
        assert all(c["messages"][-2]["role"] == "assistant" and not c["messages"][-2].get("tool_calls") for c in checks)
        work = jobs(plans, checks)
        assert len(work) == 2 * 2 * (len(plans) + len(checks))  # two models, guided and not
        assert {model for model, request, _ in work if request["kind"] == "self_check"} == {"helpdesk_r1", "lightning"}
    block = f"{SECTION_START}\nnew\n{SECTION_END}\n"
    assert with_section("# Bench\n\nold tables\n", block) == f"# Bench\n\nold tables\n\n{block}"
    assert with_section(f"# Bench\n{SECTION_START}\nold\n{SECTION_END}\n## Later\n", block) == f"# Bench\n{block}## Later\n"
