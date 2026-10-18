"""Walks the demo script (PLAN.md task 5.3) with demo/core.py and no UI: live to record what replay needs, then in
replay to check that every step comes back exactly as recorded.

    uv run python -m demo.rehearse --mode record                        # live, both vLLM servers up
    uv run python -m demo.rehearse --mode record --step helpdesk-only   # a new take of one step
    uv run python -m demo.rehearse --mode replay                        # no server: every step as recorded

The steps, each the team and single_base on the demo request: helpdesk-only (the ticket works, and the plan says no
active specialist can roll back), both (ticket first, its ID passed to ops, pods back to Running), and chaos (both,
with chaos at --chaos-rate and --chaos-seed: the seatbelt's automatic retries, and any escalation). Each step records
under its own cache tag (demo/core.py), so a new take of one leaves the others. Lightning samples at temperature 0.6,
so a take can miss the demo's point (a helpdesk-only plan once left the rollback out of `uncovered`): take it again
until it shows it. A take recorded on the demo page (Models: record) with a step's specialists and chaos switch
counts as that step's take too.

runs/demo/rehearsal_record.json keeps each step's latest take: its settings, and each side's plan, calls, counters and
verdict. --mode replay replays every recorded step with its own settings, writes rehearsal_replay.json, and exits
non-zero if a step wasn't recorded or came back differently.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

from demo import core
from eval.run_eval import REPO, git

OUT = REPO / "runs" / "demo"
RECORDING = OUT / "rehearsal_record.json"
STEPS = {"helpdesk-only": ("helpdesk only", ("helpdesk",), False), "both": ("both specialists", ("helpdesk", "ops"), False),
         "chaos": ("both, with chaos", ("helpdesk", "ops"), True)}  # --step key: (name, active specialists, chaos)
COMPARED = ("plan_steps", "calls", "answer", "passed", "escalated", "retries")


def outcome(side: core.Side) -> dict[str, Any]:
    return {"error": side.error, "verdict": side.verdict, "counters": side.counters, **side.numbers}


def entry(name: str, active: list[str], chaos_rate: float, chaos_seed: int, team: core.Side,
          single: core.Side) -> dict[str, Any]:
    return {"step": name, "active": list(active), "chaos_rate": chaos_rate, "chaos_seed": chaos_seed,
            "team": outcome(team), "single_base": outcome(single)}


def step_for(active: list[str], chaos: bool) -> str | None:
    """The demo script step that these settings play, if any."""
    for name, step_active, step_chaos in STEPS.values():
        if sorted(step_active) == sorted(active) and step_chaos == chaos:
            return name
    return None


def take(name: str, active: list[str], chaos_rate: float, chaos_seed: int, mode: str) -> dict[str, Any]:
    team, single = core.run(core.demo_case().request, list(active), chaos_rate, mode, chaos_seed=chaos_seed)
    for title, side in (("team", team), ("single_base", single)):
        uncovered = side.numbers.get("uncovered")
        print(f"{name:<18} {title:<12} " + (f"ERROR {side.error}" if side.error else
              f"{side.verdict[:1] or '-'} {side.counters.replace('**', '')}"
              + (f" · not covered: {'; '.join(uncovered)}" if uncovered else "")))
    return entry(name, active, chaos_rate, chaos_seed, team, single)


def recorded() -> list[dict[str, Any]]:
    return json.loads(RECORDING.read_text())["steps"] if RECORDING.exists() else []


def merge(steps: list[dict[str, Any]], takes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`steps` with these takes in place of their old ones, in the demo script's order."""
    by_name = {step["step"]: step for step in steps} | {step["step"]: step for step in takes}
    return [by_name[name] for name, _, _ in STEPS.values() if name in by_name]


def write(path: Any, steps: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "git_commit": git("rev-parse", "HEAD").strip(),
                                "steps": steps}, indent=2))


def save(takes: list[dict[str, Any]]) -> None:
    """Keeps these takes as their steps' recordings (demo/app.py calls it for takes recorded on the page)."""
    write(RECORDING, merge(recorded(), takes))


def differences(recording: list[dict[str, Any]], replayed: list[dict[str, Any]]) -> list[str]:
    """What didn't replay as recorded: errors (not recorded), and any compared field that changed."""
    found = []
    by_name = {step["step"]: step for step in recording}
    for now in replayed:
        was = by_name[now["step"]]
        for side in ("team", "single_base"):
            if now[side]["error"]:
                found.append(f"{now['step']}, {side}: {now[side]['error']}")
                continue
            changed = [key for key in COMPARED if was[side].get(key) != now[side].get(key)]
            if changed:
                found.append(f"{now['step']}, {side}: {', '.join(changed)} changed")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Rehearse the demo script: record it live, or check its replay.")
    parser.add_argument("--mode", choices=["record", "replay"], required=True)
    parser.add_argument("--step", choices=list(STEPS), help="only this step of the demo script")
    parser.add_argument("--chaos-rate", type=float, default=core.CHAOS_RATE, help="record: the chaos step's rate")
    parser.add_argument("--chaos-seed", type=int, default=core.CHAOS_SEED, help="record: the chaos seed")
    args = parser.parse_args()
    keys = [args.step] if args.step else list(STEPS)
    if args.mode == "record":
        takes = [take(name, list(active), args.chaos_rate if chaos else 0.0, args.chaos_seed, "record")
                 for name, active, chaos in (STEPS[key] for key in keys)]
        save(takes)
        failed = [f"{t['step']}, {side}" for t in takes for side in ("team", "single_base") if t[side]["error"]]
        print(f"Recorded {'the step' if args.step else 'every step'}." if not failed else f"These failed live: {failed}")
        sys.exit(1 if failed else 0)

    recording = recorded()
    names = [STEPS[key][0] for key in keys]
    missing = [name for name in names if name not in {step["step"] for step in recording}]
    replayed = [take(s["step"], s["active"], s["chaos_rate"], s["chaos_seed"], "replay")
                for s in recording if s["step"] in names]
    write(OUT / "rehearsal_replay.json", replayed)
    found = [f"{name}: not recorded yet" for name in missing] + differences(recording, replayed)
    print("Every step replays as recorded." if not found else "Replay differs:\n  " + "\n  ".join(found))
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
