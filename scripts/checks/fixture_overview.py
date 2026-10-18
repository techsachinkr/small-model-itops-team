"""Prints what a fixture holds, for writing cases (TEST_SET_GUIDE.md): its clock, the next ticket ID a new ticket gets,
each employee with their devices, the tickets, and each namespace's deployments with their revisions (current one
marked, healthy or broken). Reads only gym_env/fixtures/.

    .venv/bin/python scripts/checks/fixture_overview.py [fixture ...]   # no argument: list the fixtures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gym_env.state import fixture_names, load_fixture  # noqa: E402


def overview(name: str) -> list[str]:
    state = load_fixture(name)
    lines = [f"== {name}: now {state.now}; the next new ticket is HD-{state.counters['ticket']}"]
    devices: dict[str, list[str]] = {}
    for device in state.devices:
        devices.setdefault(device.employee_id, []).append(f"{device.id} {device.type} {device.model}")
    lines.append("Employees (id, name, department: devices):")
    for e in state.employees:
        lines.append(f"  {e.id} {e.name}, {e.department}: {'; '.join(devices.get(e.id, [])) or 'no devices'}")
    lines.append("Tickets (id, employee, category, priority, status, device: summary; notes):")
    for t in state.tickets:
        notes = f"; {len(t.notes)} note(s)" if t.notes else ""
        lines.append(f"  {t.id} {t.employee_id} {t.category} {t.priority} {t.status} {t.device_id or '-'}: {t.summary}{notes}")
    for namespace, deployments in state.namespaces.items():
        lines.append(f"Deployments in {namespace} (replicas; revisions, * current, x broken):")
        for d in deployments:
            revisions = ", ".join(f"{'*' if r.revision == d.current_revision else ''}{r.revision}{'' if r.healthy else 'x'} "
                                  f"{r.image.rsplit(':', 1)[-1]} ({r.deployed_at[:16]})" for r in d.revisions)
            lines.append(f"  {d.name}: {d.replicas} replicas; {revisions}")
    return lines


def main() -> None:
    names = sys.argv[1:]
    if not names:
        print("Fixtures:", ", ".join(fixture_names()))
        return
    for name in names:
        print("\n".join(overview(name)))


if __name__ == "__main__":
    main()
