"""Builds the canned ops logs from Loghub samples, the seeded incident fixtures and the ticket fixtures.

Inputs: gym_env/fixtures/default.json (hand-written) and the Loghub samples in data/open/loghub/.
Outputs, next to default.json:
- logs.json: canned log lines per revision image. Healthy images get OpenStack API request lines;
  broken ones get a start line followed by Zookeeper or Hadoop connection failures. Lines keep
  Loghub's content and get new timestamps just before the fixture's NOW. They keep Loghub's license.
- incident_<namespace>_<deployment>.json: a copy of default with a bad deploy this morning.
- one file per entry in ticket_fixtures: a copy of default or of an incident fixture with hand-written
  helpdesk tickets added (configs/fixtures.yaml), their IDs continuing the base's counter.

    uv run python -m gym_env.fixtures.make_fixtures --config configs/fixtures.yaml
"""

import argparse
import csv
import json
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from gym_env.state import (
    FIXTURES_DIR,
    Category,
    Deployment,
    Namespace,
    Priority,
    Revision,
    State,
    Ticket,
    TicketStatus,
)

LOGHUB_URL = "https://github.com/logpai/loghub"

# Loghub event templates used for broken revisions, per failure mode.
FAILURES = {
    "zookeeper": (
        "Zookeeper_2k.log_structured.csv",
        ("Connection broken for id", "caught end of stream exception", "Cannot open channel to",
         "Unexpected exception causing shutdown"),
    ),
    "hadoop": (
        "Hadoop_2k.log_structured.csv",
        ("ERROR IN CONTACTING RM", "Failed to renew lease", "NoRouteToHostException",
         "DataStreamer Exception", "Error Recovery for block"),
    ),
}
HEALTHY = ("OpenStack_2k.log_structured.csv",
           ("nova.osapi_compute.wsgi.server", "nova.metadata.wsgi.server"))

CHANGE_CAUSES = (
    "connection pool tuning", "client library upgrade", "config format migration",
    "new caching layer", "dependency upgrades", "startup health checks",
)


class Incident(BaseModel):
    model_config = ConfigDict(extra="forbid")

    namespace: Namespace
    deployment: str
    mode: Literal["bad_deploy", "double_bad_deploy"]


class NewTicket(BaseModel):
    """A hand-written ticket for a ticket fixture; its ID comes from the fixture's counter."""

    model_config = ConfigDict(extra="forbid")

    employee_id: str
    category: Category
    summary: str
    priority: Priority
    status: TicketStatus
    device_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class TicketFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base: str  # default or a generated incident fixture
    tickets: list[NewTicket] = Field(min_length=1)


class FixturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    loghub_dir: str
    loghub_commit: str
    healthy_log_lines: int
    broken_log_lines: int
    demo_incident: Incident
    seeded_incidents: int
    ticket_fixtures: list[TicketFixture] = Field(default_factory=list)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _bump(tag: str, part: Literal["minor", "patch"]) -> str:
    major, minor, patch, rc = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?", tag).groups()
    if rc is not None:
        return f"{major}.{minor}.{patch}-rc.{int(rc) + 1}"
    if part == "minor":
        return f"{major}.{int(minor) + 1}.0"
    return f"{major}.{minor}.{int(patch) + 1}"


def _find(state: State, namespace: str, name: str) -> Deployment:
    return next(d for d in state.namespaces[namespace] if d.name == name)


def make_incident(base: State, incident: Incident, rng: random.Random) -> tuple[State, str]:
    """A copy of base where a bad deploy this morning broke one deployment."""
    state = base.model_copy(deep=True)
    state.fixture = f"incident_{incident.namespace}_{incident.deployment}"
    deployment = _find(state, incident.namespace, incident.deployment)
    latest = max(deployment.revisions, key=lambda r: r.revision)
    repo, tag = latest.image.rsplit(":", 1)
    today = _parse(state.now).replace(hour=0, minute=0)
    morning = today + timedelta(hours=7, minutes=rng.randrange(0, 50))

    plan: list[tuple[str, datetime, str]] = []  # (tag, deployed_at, change cause)
    if incident.mode == "bad_deploy":
        new_tag = _bump(tag, "minor")
        plan.append((new_tag, morning, f"Deploy {new_tag}: {rng.choice(CHANGE_CAUSES)}"))
    else:
        first = _bump(tag, "minor")
        yesterday = today - timedelta(days=3 if today.weekday() == 0 else 1)  # Monday: last Friday
        plan.append((first, yesterday + timedelta(hours=17, minutes=rng.randrange(0, 50)),
                     f"Deploy {first}: {rng.choice(CHANGE_CAUSES)}"))
        second = _bump(first, "patch")
        plan.append((second, morning, f"Deploy {second}: hotfix for startup crash"))

    for offset, (new_tag, deployed_at, cause) in enumerate(plan, start=1):
        deployment.revisions.append(
            Revision(revision=latest.revision + offset, image=f"{repo}:{new_tag}",
                     change_cause=cause, deployed_at=_iso(deployed_at), healthy=False)
        )
    deployment.current_revision = deployment.revisions[-1].revision
    description = (
        f"{incident.namespace}/{incident.deployment} is broken: current revision "
        f"{deployment.current_revision} ({deployment.current.image}, deployed {deployment.current.deployed_at})"
        f" crash-loops. The last healthy revision is {latest.revision} ({latest.image})."
    )
    return state, description


def add_tickets(base: State, spec: TicketFixture) -> tuple[State, str]:
    """A copy of base with spec's tickets added in order, their IDs from base's counter."""
    state = base.model_copy(deep=True)
    state.fixture = spec.name
    employees = {employee.id for employee in state.employees}
    devices = {device.id: device.employee_id for device in state.devices}
    added = []
    for new in spec.tickets:
        if new.employee_id not in employees:
            raise ValueError(f"{spec.name}: unknown employee {new.employee_id}")
        if new.device_id is not None and devices.get(new.device_id) != new.employee_id:
            raise ValueError(f"{spec.name}: device {new.device_id} isn't assigned to {new.employee_id}")
        ticket = Ticket(id=state.next_id("ticket", "HD"), **new.model_dump())
        state.tickets.append(ticket)
        added.append(f"{ticket.id} ({ticket.employee_id}, {ticket.category} {ticket.priority}, {ticket.status})")
    description = f"{base.fixture} plus {len(added)} tickets: {', '.join(added)}."
    return state, description


def _read(loghub_dir: Path, filename: str) -> list[dict[str, str]]:
    with open(loghub_dir / filename, newline="") as f:
        return list(csv.DictReader(f))


def _format(moment: datetime, row: dict[str, str]) -> str:
    stamp = moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"
    return f"{stamp} {row['Level']:<5} {row['Component']}  {row['Content']}"


def _thin(rows: list[dict[str, str]], per_template: int = 20) -> list[dict[str, str]]:
    """Keep at most per_template evenly spaced rows of each Loghub event template, in their original
    order, so one frequent template doesn't fill every log."""
    by_template: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_template.setdefault(row["EventTemplate"], []).append(row)
    kept = {id(r) for group in by_template.values() for r in group[:: -(-len(group) // per_template)]}
    return [row for row in rows if id(row) in kept]


def _spread(pool: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    """count rows sampled across the pool, in their original order."""
    return [pool[i] for i in sorted(rng.sample(range(len(pool)), count))]


def _timestamps(now: datetime, count: int, rng: random.Random) -> list[datetime]:
    moments, moment = [], now - timedelta(seconds=rng.uniform(1, 20))
    for _ in range(count):
        moments.append(moment)
        moment -= timedelta(seconds=rng.uniform(0.3, 25))
    return moments[::-1]


def make_logs(images: dict[str, bool], now: datetime, config: FixturesConfig) -> dict[str, list[str]]:
    loghub_dir = Path(config.loghub_dir)
    filename, components = HEALTHY
    healthy_pool = _thin(
        [r for r in _read(loghub_dir, filename) if r["Level"] == "INFO" and r["Component"] in components]
    )
    failure_pools = {
        mode: _thin(
            [r for r in _read(loghub_dir, filename) if r["Level"] != "INFO" and any(k in r["Content"] for k in keys)]
        )
        for mode, (filename, keys) in FAILURES.items()
    }
    logs = {}
    for image, healthy in sorted(images.items()):
        rng = random.Random(f"{config.seed}:{image}")
        if healthy:
            rows = _spread(healthy_pool, config.healthy_log_lines, rng)
        else:
            pool = failure_pools[rng.choice(sorted(failure_pools))]
            name, version = image.rsplit("/", 1)[-1].split(":")
            rows = [{"Level": "INFO", "Component": "main", "Content": f"Starting {name} {version}"}]
            rows += _spread(pool, config.broken_log_lines - 1, rng)
        logs[image] = [_format(m, r) for m, r in zip(_timestamps(now, len(rows), rng), rows)]
    return logs


def generate(config: FixturesConfig, out_dir: Path) -> list[Path]:
    base = State.model_validate(json.loads((FIXTURES_DIR / "default.json").read_text()))
    config_dump = config.model_dump()
    rng = random.Random(config.seed)

    candidates = [
        Incident(namespace=ns, deployment=d.name, mode="bad_deploy")
        for ns, deployments in base.namespaces.items() for d in deployments
        if (ns, d.name) != (config.demo_incident.namespace, config.demo_incident.deployment)
    ]
    modes = ("bad_deploy", "double_bad_deploy")
    incidents = [config.demo_incident] + [
        c.model_copy(update={"mode": modes[i % 2]})
        for i, c in enumerate(rng.sample(candidates, config.seeded_incidents))
    ]

    # Each generated fixture with what its generated_by records besides the script and config.
    fixtures: list[tuple[State, dict[str, Any]]] = [(base, {})]
    for incident in incidents:
        state, description = make_incident(base, incident, rng)
        fixtures.append((state, {"base": "default", "incident": incident.model_dump(), "description": description}))
    for spec in config.ticket_fixtures:
        bases = {state.fixture: state for state, _ in fixtures}
        if spec.base not in bases:
            raise ValueError(f"{spec.name}: base fixture {spec.base!r} isn't generated before it")
        state, description = add_tickets(bases[spec.base], spec)
        fixtures.append((state, {"base": spec.base, "description": description}))

    images: dict[str, bool] = {}
    for state, _ in fixtures:
        for deployments in state.namespaces.values():
            for revision in (r for d in deployments for r in d.revisions):
                if images.setdefault(revision.image, revision.healthy) != revision.healthy:
                    raise ValueError(f"{revision.image} is both healthy and broken")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for state, generated_by in fixtures[1:]:
        data = {
            "generated_by": {"script": "gym_env/fixtures/make_fixtures.py", "config": config_dump, **generated_by},
            **state.model_dump(mode="json", exclude_none=True),
        }
        path = out_dir / f"{state.fixture}.json"
        path.write_text(json.dumps(data, indent=1) + "\n")
        written.append(path)
    for stale in set(out_dir.glob("incident_*.json")) - set(written):
        stale.unlink()

    logs = {
        "generated_by": {
            "script": "gym_env/fixtures/make_fixtures.py", "config": config_dump,
            "loghub": {"repo": LOGHUB_URL, "commit": config.loghub_commit,
                       "files": sorted({HEALTHY[0], *(f for f, _ in FAILURES.values())}),
                       "license": "Loghub's license (research or academic use; cite the Loghub paper). "
                                  "Log line content comes from Loghub; timestamps are ours."},
        },
        "logs": make_logs(images, _parse(base.now), config),
    }
    path = out_dir / "logs.json"
    path.write_text(json.dumps(logs, indent=1) + "\n")
    written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/fixtures.yaml")
    parser.add_argument("--out-dir", default=str(FIXTURES_DIR))
    args = parser.parse_args()
    config = FixturesConfig.model_validate(yaml.safe_load(Path(args.config).read_text()))
    for path in generate(config, Path(args.out_dir)):
        print(path)


if __name__ == "__main__":
    main()
