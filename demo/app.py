"""The demo (task 5.3): a Gradio page on demo/core.py.

    uv run --group demo python -m demo.app [--host 127.0.0.1] [--port 7860]

The Live tab has a request box with the demo request; the active specialists, with only helpdesk ticked at first as
the demo script starts (agents/registry.json isn't touched, so the evals keep every specialist); chaos on or off,
with its rate and seed; and models live, live and recorded (a good take, saved for replay), or replayed from runs/cache/ with no
server, the fallback (demo/rehearse.py records every step of the demo script). Each run shows the team (team_escalate) and single_base side by side: Lightning's plan,
each step with its tool calls, seatbelt events and any escalation, what changed in the environment, the answer, the
verifier's verdict, and two counters (steps handled by small models, and the request's cost). The Results tab shows
the specialist ladder of each API, the system table and the vLLM serving results, as eval/report.py and bench/run.py wrote
them. The page listens on 127.0.0.1; reach it through an SSH tunnel (ssh -L 7860:localhost:7860 <box>).
"""

import argparse

import gradio as gr

from demo import core, rehearse

PANELS = 5  # per side: title and counters, plan and steps, state change, answer, verdict


def panels(side: core.Side) -> list[str]:
    if side.error:
        return [f"### {side.title}", f"⚠️ {side.error}", "", "", ""]
    return [f"### {side.title}\n{side.counters}", side.detail, f"**What changed in the environment**\n\n{side.state}",
            f"**Answer**\n\n{side.answer}", side.verdict]


def run(request: str, active: list[str], chaos: bool, rate: float, seed: float, mode: str) -> list[str]:
    rate = rate if chaos else 0.0
    team, single = core.run(request, list(active), rate, mode, chaos_seed=int(seed))
    step = rehearse.step_for(list(active), chaos)
    if mode == "record" and step and request.strip() == core.demo_case().request and not (team.error or single.error):
        # a take of a demo script step: replay checks (demo/rehearse.py, scripts/demo_reset.sh) expect this one now
        rehearse.save([rehearse.entry(step, list(active), rate, int(seed), team, single)])
    return [*panels(team), *panels(single)]


def ladder(api: str) -> "object | None":
    import pandas as pd  # a Gradio dependency

    rows = [row for row in core.ladder_rows() if row["api"] == api]
    return pd.DataFrame(rows) if rows else None


def build() -> gr.Blocks:
    case = core.demo_case()
    with gr.Blocks(title="Small-model IT ops team") as page:
        gr.Markdown("# 🧭 Small-model IT ops team\nNemotron 3.5 Lightning plans and steps in when needed; small specialists "
                    "trained with NeMo Gym on its verified work make the tool calls. Every run starts from the same "
                    f"payments incident, with {case.user_id} as the acting user. Costs are GPU time at full load "
                    "(configs/eval.yaml); adapters are priced as the base model although LoRA serving is slower (task "
                    "4.1), so the team's cost is understated next to single_base's.")
        with gr.Tab("Live"):
            request = gr.Textbox(case.request, label="Request", lines=2)
            with gr.Row():
                active = gr.CheckboxGroup([(core.label(name), name) for name in core.SPECIALISTS], value=["helpdesk"],
                                          label="Active specialists")
                chaos = gr.Checkbox(False, label="Chaos: tools answer 503 at random")
                rate = gr.Slider(0.05, 0.9, value=core.CHAOS_RATE, step=0.05, label="Chaos rate")
                seed = gr.Number(core.CHAOS_SEED, precision=0, label="Chaos seed")
                mode = gr.Radio(["live", "record", "replay"], value="live",
                                label="Models (record: live, saving the answers for replay)")
            button = gr.Button("Run", variant="primary")
            with gr.Row():
                with gr.Column():
                    team = [gr.Markdown() for _ in range(PANELS)]
                with gr.Column():
                    single = [gr.Markdown() for _ in range(PANELS)]
            button.click(run, [request, active, chaos, rate, seed, mode], [*team, *single], concurrency_limit=1)
        with gr.Tab("Results"):
            with gr.Row():
                for api in core.SPECIALISTS:
                    frame = ladder(api)
                    if frame is not None:
                        gr.BarPlot(frame, x="setup", y="cases passed", sort=None,
                                   title=f"{core.label(api)} specialist ladder: practice cases passed, of 12")
            for name, text in core.result_tables().items():
                with gr.Accordion(name, open=name == "system"):
                    gr.Markdown(text)
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description="The demo page.")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to listen on every interface")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build().launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
