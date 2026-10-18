"""demo/core.py without servers: scripted models play the team (team_escalate) and single_base on the demo request,
each against its own fresh fixture, and the panels show the plan, steps, escalations, state change and counters."""

import json
from typing import Any

import pytest

from common.eval_config import load_eval_config
from common.tool_loop import ModelReply, ToolCall
from demo import core

TICKET = {"employee_id": "E1012", "category": "incident", "summary": "Payments crashing in prod after the deploy",
          "priority": "P1"}
ROLLBACK = {"namespace": "prod", "deployment": "payments", "to_revision": 8, "change_ticket": "HD-1042"}
REQUEST = core.demo_case().request


def turn(*calls: tuple[str, Any], text: str = "") -> ModelReply:
    tool_calls = [ToolCall(f"call-{i}", name, json.dumps(args)) for i, (name, args) in enumerate(calls)]
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                                 for c in tool_calls]
    return ModelReply(message, text, tool_calls, prompt_tokens=1000, completion_tokens=50, latency_s=0.0)


def done(text: str) -> list[ModelReply]:
    return [turn(text=text), turn(text=json.dumps({"reason": "done", "tool_call_needed": False}))]


def plan(*steps: tuple[str, str, list[int]], uncovered: tuple[str, ...] = ()) -> ModelReply:
    return turn(text=json.dumps({"analysis": "the user wants a rollback and a ticket", "uncovered": list(uncovered),
                                 "reply": "", "steps": [{"agent": a, "instruction": i, "depends_on": d} for a, i, d in steps]}))


DEMO_PLAN = plan(("helpdesk", "Open a P1 incident ticket for E1012: payments in prod crashes after the deploy.", []),
                 ("ops", "Roll payments in prod back to its last healthy revision, with step 1's ticket as change_ticket.", [1]))


class Scripted:
    """One scripted model per name; the planner and escalation share "lightning", in the order they're asked."""

    def __init__(self, **scripts: list[ModelReply]) -> None:
        self.scripts = {name: list(replies) for name, replies in scripts.items()}
        self.asked: list[str] = []

    def __call__(self, name: str):
        async def model(messages: Any, tools: Any, schema: Any) -> ModelReply:
            self.asked.append(name)
            return self.scripts[name].pop(0)

        return model


def ops_rolls_back() -> list[ModelReply]:
    return [turn(("rollout_history", {"namespace": "prod", "deployment": "payments"})), turn(("rollback", ROLLBACK)),
            *done("Rolled payments back to revision 8.")]


def single_base_fumbles() -> list[ModelReply]:
    """Rolls back without a change ticket (the environment refuses it), then claims it's done."""
    no_ticket = {k: v for k, v in ROLLBACK.items() if k != "change_ticket"}
    return [turn(("rollback", no_ticket)), *done("I rolled payments back.")]


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """configs/eval.yaml's seatbelt, without waiting between automatic 503 retries."""
    config = load_eval_config()
    fast = config.model_copy(update={"seatbelt": config.seatbelt.model_copy(update={"transient_backoff_s": 0})})
    monkeypatch.setattr(core, "load_eval_config", lambda: fast)


def test_the_demo_request_on_the_team_and_on_single_base() -> None:
    models = Scripted(lightning=[DEMO_PLAN], helpdesk_r1=[turn(("create_ticket", TICKET)), *done("Opened HD-1042.")],
                      ops_r1=ops_rolls_back(), student=single_base_fumbles())
    team, single = core.run(REQUEST, ["helpdesk", "ops"], models=models)

    assert team.error is None and single.error is None
    assert "🧭 Lightning's plan" in team.detail and "🎫 Helpdesk" in team.detail and "🛠️ Ops (after 1)" in team.detail
    assert "`create_ticket`" in team.detail and "→ 200" in team.detail
    assert "Opened **HD-1042**: P1 incident for E1012" in team.state
    assert "Rollback **prod/payments**: revision 9 → 8 with change ticket HD-1042" in team.state
    assert "→ **4 Running**" in team.state  # the incident's pods were crashing before
    assert team.verdict.startswith("✅") and team.counters.startswith("**2/2** steps handled by small models")
    assert team.numbers["cost_usd"] > 0 and team.numbers["escalated"] == 0

    assert "→ 400" in single.detail and single.state == "- Nothing changed." and single.verdict.startswith("❌")
    assert "**1/1** steps by a small model" in single.counters


def test_with_only_helpdesk_active_the_plan_says_what_isnt_covered() -> None:
    models = Scripted(lightning=[plan(("helpdesk", "Open a P1 incident ticket for E1012 about payments.", []),
                                      uncovered=("Roll back the payments deployment in prod",))],
                      helpdesk_r1=[turn(("create_ticket", TICKET)), *done("Opened HD-1042.")],
                      student=single_base_fumbles())
    team, _ = core.run(REQUEST, ["helpdesk"], models=models)
    assert "Not covered by the active specialists: Roll back the payments deployment in prod" in team.detail
    assert team.numbers["uncovered"] == ["Roll back the payments deployment in prod"]
    assert "🛠️ Ops" not in team.detail and "Opened **HD-1042**" in team.state and "Rollback" not in team.state
    assert team.verdict.startswith("❌") and team.counters.startswith("**1/1**")


def test_notes_on_a_ticket_the_run_opened_are_shown() -> None:
    """A note can claim what never happened (a helpdesk-only take noted a rollback to "revision 2.1.3")."""
    note = {"ticket_id": "HD-1042", "note": "Payments app rolled back to revision 2.1.3 after morning deploy."}
    models = Scripted(lightning=[plan(("helpdesk", "Open a P1 incident ticket for E1012 about payments.", []),
                                      ("helpdesk", "Add a note saying which revision payments was rolled back to.", [1]))],
                      helpdesk_r1=[turn(("create_ticket", TICKET)), *done("Opened HD-1042."), turn(("update_ticket", note)),
                                   *done("Noted.")],
                      student=single_base_fumbles())
    team, _ = core.run(REQUEST, ["helpdesk"], models=models)
    assert "; note: “Payments app rolled back to revision 2.1.3 after morning deploy.”" in team.state
    assert "Rollback" not in team.state


def test_a_step_the_seatbelt_gives_up_on_is_shown_taken_over_by_lightning() -> None:
    models = Scripted(lightning=[DEMO_PLAN, *done("HD-1042 is open for the payments incident.")],
                      helpdesk_r1=[turn(("create_ticket", TICKET)), turn(*[("drop_tickets", {})] * 3)],
                      ops_r1=ops_rolls_back(), student=single_base_fumbles())
    team, _ = core.run(REQUEST, ["helpdesk", "ops"], models=models)
    assert "**⚡ Lightning took the step over** on `lightning`" in team.detail
    assert "seatbelt gave up" in team.detail and team.counters.startswith("**1/2** steps handled by small models")
    assert team.verdict.startswith("✅") and team.numbers["escalated"] == 1


def test_chaos_shows_the_automatic_retries_and_is_the_same_every_time() -> None:
    def scripts() -> Scripted:
        return Scripted(lightning=[DEMO_PLAN], helpdesk_r1=[turn(("create_ticket", TICKET)), *done("Opened HD-1042.")],
                        ops_r1=ops_rolls_back(), student=single_base_fumbles())

    first, _ = core.run(REQUEST, ["helpdesk", "ops"], chaos_rate=core.CHAOS_RATE, models=scripts())
    again, _ = core.run(REQUEST, ["helpdesk", "ops"], chaos_rate=core.CHAOS_RATE, models=scripts())
    assert first.numbers["retries"] > 0 and "transient retry" in first.detail
    assert first.detail == again.detail  # chaos is seeded by the case: a replay repeats it


def test_replay_without_a_recording_shows_an_error_instead_of_crashing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from common import llm

    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    team, single = core.run(REQUEST, ["helpdesk", "ops"], mode="replay")
    assert "Not recorded for replay" in team.error and "Not recorded for replay" in single.error


def test_each_scenario_records_apart_and_the_environment_is_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    assert core.cache_tag(["ops", "helpdesk"], 0.0) == "demo:helpdesk+ops:chaos=0:seed=5" != core.cache_tag(["helpdesk", "ops"], 0.3)
    monkeypatch.setenv("LLM_MODE", "live")
    monkeypatch.delenv("LLM_CACHE_TAG", raising=False)
    models = Scripted(lightning=[DEMO_PLAN], helpdesk_r1=[turn(("create_ticket", TICKET)), *done("Opened HD-1042.")],
                      ops_r1=ops_rolls_back(), student=single_base_fumbles())
    core.run(REQUEST, ["helpdesk", "ops"], mode="record", models=models)
    import os

    assert os.environ["LLM_MODE"] == "live" and "LLM_CACHE_TAG" not in os.environ


def test_the_ladder_rows_come_from_the_report_table() -> None:
    rows = core.ladder_rows()
    if not rows:
        pytest.skip("runs/practice_ladder_r1.md isn't there (runs/ is gitignored)")
    ops = {row["setup"]: row["cases passed"] for row in rows if row["api"] == "ops"}
    assert {"base_raw", "base_seatbelt", "r1", "lightning"} <= ops.keys() and all(row["cases"] == 12 for row in rows)


def test_the_rehearsal_reports_what_didnt_replay_as_recorded() -> None:
    from demo import rehearse

    side = {"error": None, "plan_steps": ["helpdesk", "ops"], "calls": ["create_ticket"], "answer": "Opened HD-1042.",
            "passed": True, "escalated": 0, "retries": 0}
    step = {"step": "both specialists", "chaos_rate": 0.0, "chaos_seed": 5, "team": side, "single_base": side}
    assert rehearse.differences([step], [step]) == []
    missed = {**step, "team": {**side, "error": "Not recorded for replay"}}
    changed = {**step, "single_base": {**side, "calls": ["rollback"], "passed": False}}
    assert rehearse.differences([step], [missed]) == ["both specialists, team: Not recorded for replay"]
    assert rehearse.differences([step], [changed]) == ["both specialists, single_base: calls, passed changed"]
    chaos = {**step, "step": "both, with chaos", "chaos_rate": 0.3}
    retake = {**step, "team": {**side, "answer": "a new take"}}
    assert rehearse.merge([chaos, step], [retake]) == [retake, chaos]  # the demo script's order, the new take in place


def test_settings_map_to_the_demo_script_s_steps() -> None:
    from demo import rehearse

    assert rehearse.step_for(["helpdesk"], False) == "helpdesk only"
    assert rehearse.step_for(["ops", "helpdesk"], False) == "both specialists"
    assert rehearse.step_for(["helpdesk", "ops"], True) == "both, with chaos"
    assert rehearse.step_for(["ops"], False) is None and rehearse.step_for(["helpdesk"], True) is None


def test_a_take_recorded_on_the_page_becomes_its_step_s_recording(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    pytest.importorskip("gradio")  # the demo dependency group
    from demo import app, rehearse

    monkeypatch.setattr(rehearse, "OUT", tmp_path)
    monkeypatch.setattr(rehearse, "RECORDING", tmp_path / "rehearsal_record.json")
    sides = (core.Side(title="team", verdict="✅", numbers={"answer": "a"}), core.Side(title="single", numbers={}))
    monkeypatch.setattr(core, "run", lambda *args, **kwargs: sides)
    app.run(REQUEST, ["helpdesk"], False, 0.3, 5, "live")
    assert rehearse.recorded() == []  # only takes recorded for replay count
    app.run(REQUEST, ["helpdesk"], False, 0.3, 5, "record")
    (step,) = rehearse.recorded()
    assert (step["step"], step["active"], step["chaos_rate"], step["team"]["answer"]) == ("helpdesk only", ["helpdesk"], 0.0, "a")
    app.run("Something else entirely.", ["helpdesk", "ops"], False, 0.3, 5, "record")
    assert [s["step"] for s in rehearse.recorded()] == ["helpdesk only"]  # not the demo request: not a demo step's take


def test_the_page_builds_with_the_demo_s_controls() -> None:
    pytest.importorskip("gradio")  # the demo dependency group
    from demo import app

    page = app.build()
    labels = {getattr(block, "label", None) for block in page.blocks.values()}
    assert {"Request", "Active specialists", "Chaos rate", "Chaos seed"} <= labels
    assert any(str(label).startswith("Models") for label in labels)
    team = app.panels(core.Side(title="t", counters="c", detail="d", state="s", answer="a", verdict="v"))
    assert len(team) == app.PANELS and app.panels(core.Side(title="t", error="down"))[1] == "⚠️ down"
