"""Per-task environment state: fixture loading, ID counters and the frozen clock."""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field

from common.schemas import Strict

FIXTURES_DIR = Path(__file__).parent / "fixtures"

Namespace = Literal["prod", "staging"]
Category = Literal["hardware", "software", "access", "network", "incident"]
Priority = Literal["P1", "P2", "P3", "P4"]
TicketStatus = Literal["open", "in_progress", "resolved"]
Delivery = Literal["email", "sms"]


class Employee(Strict):
    id: str
    name: str
    email: str
    department: str


class Device(Strict):
    id: str
    employee_id: str
    type: Literal["laptop", "monitor", "phone", "dock"]
    model: str


class Ticket(Strict):
    id: str
    employee_id: str
    category: Category
    summary: str
    priority: Priority
    status: TicketStatus
    device_id: str | None = None
    notes: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class PasswordReset(Strict):
    employee_id: str
    delivery: Delivery
    at: str


class Revision(Strict):
    revision: int
    image: str
    change_cause: str
    deployed_at: str
    healthy: bool  # never shown to the model; pods and logs follow it


class Deployment(Strict):
    name: str
    replicas: int
    current_revision: int
    revisions: list[Revision]

    def revision(self, number: int) -> Revision | None:
        return next((r for r in self.revisions if r.revision == number), None)

    @property
    def current(self) -> Revision:
        revision = self.revision(self.current_revision)
        assert revision is not None, f"{self.name}: current revision missing from history"
        return revision


class Change(Strict):
    """One applied change to a deployment, with the helpdesk ticket that authorized it (prod only)."""

    namespace: Namespace
    deployment: str
    action: Literal["rollback", "scale"]
    before: int
    after: int
    change_ticket: str | None
    at: str


class State(Strict):
    fixture: str
    now: str  # frozen clock, an ISO timestamp
    counters: dict[str, int]  # next number per ID table, e.g. {"ticket": 1042} -> HD-1042
    employees: list[Employee]
    devices: list[Device]
    tickets: list[Ticket]
    password_resets: list[PasswordReset] = Field(default_factory=list)
    namespaces: dict[Namespace, list[Deployment]]
    changes: list[Change] = Field(default_factory=list)

    def next_id(self, table: str, prefix: str) -> str:
        number = self.counters[table]
        self.counters[table] = number + 1
        return f"{prefix}-{number}"


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURES_DIR.glob("*.json") if p.stem != "logs")


@lru_cache(maxsize=None)
def _load(name: str) -> State:
    if not re.fullmatch(r"[a-z0-9_]+", name) or name == "logs":
        raise ValueError(f"invalid fixture name {name!r}")
    path = FIXTURES_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"unknown fixture {name!r}; available: {', '.join(fixture_names())}")
    data = json.loads(path.read_text())
    data.pop("generated_by", None)
    state = State.model_validate(data)
    if state.fixture != name:
        raise ValueError(f"{path} says fixture {state.fixture!r}")
    return state


def load_fixture(name: str) -> State:
    """A fresh, independent copy of a fixture's state."""
    return _load(name).model_copy(deep=True)
