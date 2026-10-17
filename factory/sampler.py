"""Samples valid actions from fixtures and computes their gold calls in code.

A sampled task is a Case without its request, plus a brief for factory/writer.py: what the user wants,
the facts the request must state (`include`) and the ones it must leave out (`leave_out`). Code picks
the action from the fixture's state, so the gold calls are right by construction, and a model only
phrases the request (origin=sampled). The groups, for each API:
- single_1_step: the request gives every ID the call needs.
- single_2_steps: one lookup, then the action ("my laptop" needs list_devices for the device_id, "my
  VPN ticket" needs list_tickets, "back to 2.3.1" needs rollout_history, "one more replica" needs
  list_deployments).
- missing_info: the request leaves out something the action needs (which device, which namespace, a
  change ticket, a colleague's employee ID); nothing should change.
- no_tool: a question, or a request no tool can or should carry out; nothing should change.

    uv run python -m factory.sampler --api ops --n 20    # print sampled tasks, without a model
"""

import argparse
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from common.schemas import Api, Case, GoldCall, Kind, Strict
from gym_env.state import Deployment, Device, Employee, State, Ticket, fixture_names, load_fixture
from gym_env.verify import GoldCallFailed, gold_run, project

REPO = Path(__file__).resolve().parent.parent
FACTORY_CONFIG = REPO / "configs" / "factory.yaml"
GROUPS = ("single_1_step", "single_2_steps", "missing_info", "no_tool")
GROUP_KIND: dict[str, Kind] = {
    "single_1_step": "single", "single_2_steps": "single", "missing_info": "missing_info", "no_tool": "no_tool",
}


class Mention(Strict):
    """A fact the request must state or leave out: shown to the writer as `text`, checked with `pattern`
    (a regex; case is ignored)."""

    text: str
    pattern: str


class SampledTask(Strict):
    api: Api
    group: str
    template: str
    user_id: str
    fixture: str
    gold_calls: list[GoldCall]
    brief: str  # what the user wants, in the second person ("Ask the assistant to ...")
    include: list[Mention] = Field(default_factory=list)
    leave_out: list[Mention] = Field(default_factory=list)

    @property
    def kind(self) -> Kind:
        return GROUP_KIND[self.group]

    def case(self, case_id: str, request: str) -> Case:
        return Case(id=case_id, request=request, user_id=self.user_id, fixture=self.fixture, apis=[self.api],
                    gold_calls=self.gold_calls, kind=self.kind)

    def phrasing_problems(self, request: str) -> list[str]:
        """What a phrased request gets wrong against the brief (empty if nothing)."""
        problems = [f"it must include {m.text}" for m in self.include if not re.search(m.pattern, request, re.I)]
        problems += [f"it must leave out {m.text}" for m in self.leave_out if re.search(m.pattern, request, re.I)]
        return problems


def task_problems(case: Case) -> list[str]:
    """Why a task can't be used (empty if it can). Its gold calls must run; single tasks must change the
    state and other kinds must not; a read_only task's answer strings must come from what its gold calls
    return and not from the request."""
    try:
        gold = gold_run(case)
    except GoldCallFailed as error:
        return [str(error)]
    changed = project(gold.state) != project(load_fixture(case.fixture))
    problems = []
    if case.kind in ("single", "multi") and not changed:
        problems.append("the gold calls change nothing")
    if case.kind not in ("single", "multi") and changed:
        problems.append(f"the gold calls of a {case.kind} task change the state")
    if case.kind == "read_only":
        returned = " ".join(json.dumps(result.output, ensure_ascii=False) for result in gold.calls).casefold()
        if missing := [s for s in case.answer_must_contain or [] if s.casefold() not in returned]:
            problems.append(f"answer strings no gold call returns: {missing}")
        if given := [s for s in case.answer_must_contain or [] if s.casefold() in case.request.casefold()]:
            problems.append(f"answer strings already in the request: {given}")
    return problems


# Patterns for Mention.

def exact(text: str) -> str:
    """`text` as a whole word or ID."""
    return rf"(?<![\w-]){re.escape(text)}(?![\w-])"


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve"]


def count(n: int, words: bool = True) -> str:
    word = f"|{NUMBER_WORDS[n]}" if words and n < len(NUMBER_WORDS) else ""
    return rf"(?<![\w.-])({n}{word})(?![\w-]|\.\d)"


def version(tag: str) -> str:
    return rf"(?<![\w.-])v?{re.escape(tag)}(?![\w-]|\.\d)"


NAMESPACE = {"prod": r"\bprod(uction)?\b", "staging": r"\bstag(e|ing)\b"}
NAMESPACE_TEXT = {"prod": "prod", "staging": "staging"}
# Asking for a delivery method, not just naming one: "can't access their email" doesn't ask for email delivery
# (pilot task hd-pilot-0017), but "to my email", "by email" and "email it to her" do.
DELIVERY = {
    "email": Mention(text="that it goes by email",
                     pattern=r"\b(by|via|over|through|to|on)\s+(\w+\s+){0,2}e-?mail\b|\be-?mail(ed)?\s+(it|me|him|her|them|the)\b"),
    "sms": Mention(text="that it goes by text message (SMS)",
                   pattern=r"\b(sms|texts?|texted|texting)\b|\b(to|on)\s+(\w+\s+){0,2}(phone|mobile|cell)\b"),
}
ANY_TICKET_ID = Mention(text="any ticket ID", pattern=r"\bHD-?\s*\d+")
ANY_DEVICE_ID = Mention(text="any device ID", pattern=r"\bD\d{4}\b")


def priority_mention(priority: str) -> Mention:
    return Mention(text=f"the priority, written {priority}", pattern=exact(priority))


def id_mention(what: str, value: str) -> Mention:
    return Mention(text=f"{what} {value}", pattern=exact(value))


def call(api: str, tool: str, **args: Any) -> GoldCall:
    return GoldCall(api=api, tool=tool, args={key: value for key, value in args.items() if value is not None})


# Templates. Each draws one action from the fixture's state, or returns None if the fixture has none.

Make = Callable[[random.Random, State], SampledTask | None]


@dataclass(frozen=True)
class Template:
    name: str
    api: str
    group: str
    make: Make


TEMPLATES: list[Template] = []


def template(api: str, group: str) -> Callable[[Make], Make]:
    def register(make: Make) -> Make:
        TEMPLATES.append(Template(make.__name__, api, group, make))
        return make

    return register


def templates_for(api: str, group: str) -> list[Template]:
    return [t for t in TEMPLATES if t.api == api and t.group == group]


def task(state: State, make: Make, user: Employee, gold_calls: list[GoldCall], brief: str,
         include: list[Mention] | None = None, leave_out: list[Mention] | None = None) -> SampledTask:
    spec = next(t for t in TEMPLATES if t.make is make)
    return SampledTask(api=spec.api, group=spec.group, template=spec.name, user_id=user.id, fixture=state.fixture,
                       gold_calls=gold_calls, brief=brief, include=include or [], leave_out=leave_out or [])


# Helpdesk

IT_STAFF = "E1011"  # Mei Lin, IT: the requester who acts for colleagues, by their employee ID

# (summary for the gold call, what the user tells the assistant). Each detail should name one category.
ISSUES: dict[str, list[tuple[str, str]]] = {
    "software": [
        ("Outlook crashes on startup", "Outlook crashes every time it opens, since this morning"),
        ("Request to install Tableau Desktop", "Tableau Desktop is needed on the laptop for a new reporting project"),
        ("Zoom has no audio after the latest update", "Zoom stopped playing any audio after it updated itself yesterday"),
        ("Excel budgeting add-in fails to load", "the company's Excel budgeting add-in fails to load with an error"),
        ("Request for a Microsoft Visio license", "a Microsoft Visio license is needed to draw process diagrams"),
        ("Slack desktop app freezes during screen sharing", "the Slack desktop app freezes whenever the screen is shared"),
        ("PDF reader errors on signed documents", "the PDF reader shows an error when opening signed contracts"),
    ],
    "access": [
        ("Needs access to the Finance shared drive", "access to the Finance team's shared drive is needed"),
        ("Needs access to the Salesforce reporting workspace", "access to the Salesforce reporting workspace is needed for the quarterly review"),
        ("Add to the engineering on-call group", "they need to be added to the engineering on-call group in the paging system"),
        ("Permission denied on the legal contracts folder", "opening the legal contracts folder now says 'permission denied', though it used to work"),
        ("Needs read access to the payroll reports", "read-only access to the payroll reports is needed"),
    ],
    "network": [
        ("VPN client hangs on connecting from home", "the VPN client hangs on 'connecting' when working from home"),
        ("Wi-Fi keeps dropping on the third floor", "the Wi-Fi keeps dropping on the third floor near the kitchen"),
        ("Very slow connection to the office file server", "copying files from the office file server has been extremely slow all week"),
        ("VPN connects but internal sites don't load", "the VPN connects, but no internal sites load"),
    ],
    "incident": [
        ("Company wiki down for everyone", "the company wiki has been down for everyone since about 8:30"),
        ("Shared printers on floor 2 offline for everyone", "all the shared printers on floor 2 are offline for everyone"),
        ("Expense reporting system errors for all users", "the expense reporting system shows an error page for everyone on the team"),
        ("Internal email delivery delayed for everyone", "emails between colleagues are arriving about an hour late, for everyone"),
        ("Meeting room booking system down", "the meeting room booking system won't load for anyone"),
    ],
}
PRIORITIES = {"software": ["P2", "P3", "P3", "P4"], "access": ["P2", "P3", "P3", "P4"],
              "network": ["P2", "P3", "P3", "P4"], "incident": ["P1", "P1", "P2"], "hardware": ["P2", "P3", "P3", "P4"]}
HARDWARE: dict[str, list[tuple[str, str]]] = {
    "laptop": [
        ("Laptop screen flickers", "the laptop screen flickers every few minutes"),
        ("Laptop won't charge", "the laptop stopped charging, even with a different charger"),
        ("Laptop keyboard keys stick", "several keys on the laptop keyboard stick"),
        ("Laptop overheats and shuts down", "the laptop gets very hot and shuts itself down"),
        ("Laptop webcam not detected", "the laptop's webcam isn't detected in any app"),
        ("Laptop won't wake from sleep", "the laptop won't wake from sleep without a hard reset"),
        ("Laptop USB-C ports not working", "none of the laptop's USB-C ports work"),
        ("Laptop speakers crackle", "the laptop speakers crackle at any volume"),
    ],
    "monitor": [
        ("Monitor shows no signal", "the monitor shows 'no signal' although the cable is plugged in"),
        ("Monitor has dead pixels", "the monitor has a cluster of dead pixels in the middle"),
        ("Monitor flickers at full brightness", "the monitor flickers at full brightness"),
    ],
    "phone": [
        ("Phone screen cracked", "the phone's screen cracked after a fall"),
        ("Phone won't charge", "the phone won't charge with any cable"),
        ("Phone microphone not working", "callers can't hear anything through the phone's microphone"),
    ],
    "dock": [
        ("Dock doesn't power the laptop", "the dock no longer powers the laptop"),
        ("Dock ethernet port not working", "the dock's ethernet port doesn't work"),
        ("Dock doesn't detect external monitors", "the dock doesn't detect the external monitors anymore"),
    ],
}
DEVICE_WORD = {"laptop": r"\blaptop", "monitor": r"\bmonitor", "phone": r"\bphone", "dock": r"\bdock"}
# Problems that fit both devices of a pair, for asking which device is meant.
AMBIGUOUS_PROBLEMS: dict[frozenset[str], list[str]] = {
    frozenset({"laptop", "monitor"}): ["the screen keeps flickering", "the display has a line of dead pixels"],
    frozenset({"laptop", "dock"}): ["it stopped charging", "its USB ports stopped working"],
    frozenset({"laptop", "phone"}): ["the battery drains within a couple of hours", "it won't charge anymore"],
}
NOTES = [
    ("Restarting didn't help", "restarting didn't help"),
    ("Away until Thursday; contact after that", "you're away until Thursday, so they should contact you after that"),
    ("Getting worse: now several times a day", "it's getting worse and now happens several times a day"),
    ("Available for a call after 2pm today", "you're available for a call after 2pm today"),
]
FIXES = ["it started working again after a restart", "the problem went away after the latest update",
         "a colleague sorted it out"]


def employee(state: State, employee_id: str) -> Employee:
    return next(e for e in state.employees if e.id == employee_id)


def devices_of(state: State, employee_id: str) -> list[Device]:
    return [d for d in state.devices if d.employee_id == employee_id]


def tickets_of(state: State, employee_id: str) -> list[Ticket]:
    return [t for t in state.tickets if t.employee_id == employee_id]


def requester(rng: random.Random, state: State, subjects: list[Employee]) -> tuple[Employee, Employee] | None:
    """(who asks, who it's for): usually the employee themselves, sometimes IT staff on their behalf."""
    if not subjects:
        return None
    subject = rng.choice(subjects)
    if subject.id != IT_STAFF and rng.random() < 0.25:
        return employee(state, IT_STAFF), subject
    return subject, subject


def for_whom(user: Employee, subject: Employee, own: str, theirs: str) -> tuple[str, list[Mention]]:
    """Brief wording and mentions for acting on your own behalf or for a colleague, by employee ID."""
    if user.id == subject.id:
        return own, []
    phrase = theirs.format(name=subject.name, id=subject.id)
    return phrase, [id_mention("the colleague's employee ID", subject.id)]


@template("helpdesk", "single_1_step")
def reset_password(rng: random.Random, state: State) -> SampledTask | None:
    user, subject = requester(rng, state, state.employees)
    delivery = rng.choice(["email", "sms"])
    whose, mentions = for_whom(user, subject, "your password", "the password of your colleague {name} (employee ID {id})")
    how = "by email" if delivery == "email" else "by text message"
    context = rng.choice(["", " You're locked out after too many attempts.", " The old password expired."])
    return task(state, reset_password, user, [call("helpdesk", "reset_password", employee_id=subject.id, delivery=delivery)],
                f"Ask the assistant to reset {whose} and send the temporary password {how}.{context}",
                include=[DELIVERY[delivery], *mentions])


@template("helpdesk", "single_1_step")
def ticket_without_device(rng: random.Random, state: State) -> SampledTask | None:
    user, subject = requester(rng, state, state.employees)
    category = rng.choice(list(ISSUES))
    summary, detail = rng.choice(ISSUES[category])
    priority = rng.choice(PRIORITIES[category])
    whose, mentions = for_whom(user, subject, "", " for your colleague {name} (employee ID {id})")
    return task(state, ticket_without_device, user,
                [call("helpdesk", "create_ticket", employee_id=subject.id, category=category, summary=summary, priority=priority)],
                f"Ask the assistant to open a {priority} ticket{whose}: {detail}.",
                include=[priority_mention(priority), *mentions])


def ticket_update(rng: random.Random, ticket: Ticket) -> tuple[dict[str, Any], str, list[Mention]]:
    """(update_ticket arguments, what the user asks for, mentions) for one ticket."""
    if ticket.status == "resolved":
        note, said = rng.choice([("The problem is back", "the problem is back"),
                                 ("Happened again this week", "it happened again this week")])
        return {"status": "open", "note": note}, f"reopen it and add a note that {said}", []
    action = rng.choice(["note", "priority", "resolve"])
    if action == "note":
        note, said = rng.choice(NOTES)
        return {"note": note}, f"add a note that {said}", []
    if action == "priority":
        priority = rng.choice([p for p in ("P1", "P2", "P3", "P4") if p != ticket.priority])
        why = "it's now blocking your work" if priority < ticket.priority else "you found a workaround"
        return {"priority": priority}, f"change its priority to {priority}, because {why}", [priority_mention(priority)]
    fix = rng.choice(FIXES)
    return {"status": "resolved", "note": f"Fixed: {fix}"}, f"close it, with a note that {fix}", []


def ticket_owners(state: State) -> list[Employee]:
    return [e for e in state.employees if tickets_of(state, e.id)]


@template("helpdesk", "single_1_step")
def update_ticket_by_id(rng: random.Random, state: State) -> SampledTask | None:
    pair = requester(rng, state, ticket_owners(state))
    if pair is None:
        return None
    user, subject = pair
    ticket = rng.choice(tickets_of(state, subject.id))
    args, action, mentions = ticket_update(rng, ticket)
    whose = "your" if user.id == subject.id else f"your colleague {subject.name}'s"
    return task(state, update_ticket_by_id, user, [call("helpdesk", "update_ticket", ticket_id=ticket.id, **args)],
                f"About {whose} ticket {ticket.id} (\"{ticket.summary}\", status {ticket.status}): "
                f"ask the assistant to {action}.",
                include=[id_mention("the ticket ID", ticket.id), *mentions])


@template("helpdesk", "single_2_steps")
def update_ticket_by_topic(rng: random.Random, state: State) -> SampledTask | None:
    pair = requester(rng, state, ticket_owners(state))
    if pair is None:
        return None
    user, subject = pair
    ticket = rng.choice(tickets_of(state, subject.id))
    args, action, mentions = ticket_update(rng, ticket)
    whose, who = for_whom(user, subject, "your", "your colleague {name}'s (employee ID {id})")
    return task(state, update_ticket_by_topic, user,
                [call("helpdesk", "list_tickets", employee_id=subject.id),
                 call("helpdesk", "update_ticket", ticket_id=ticket.id, **args)],
                f"About {whose} ticket on \"{ticket.summary}\" (status {ticket.status}): ask the assistant to {action}. "
                "Describe the ticket by its topic, not its ID.",
                include=[*who, *mentions], leave_out=[ANY_TICKET_ID])


def single_device_of_type(state: State, subject: Employee) -> list[Device]:
    """Devices that are the only one of their type for this employee ("my laptop" is unambiguous)."""
    devices = devices_of(state, subject.id)
    return [d for d in devices if sum(other.type == d.type for other in devices) == 1]


def hardware_ticket(rng: random.Random, state: State, make: Make, with_device_id: bool) -> SampledTask | None:
    pair = requester(rng, state, [e for e in state.employees if single_device_of_type(state, e)])
    if pair is None:
        return None
    user, subject = pair
    device = rng.choice(single_device_of_type(state, subject))
    summary, detail = rng.choice(HARDWARE[device.type])
    priority = rng.choice(PRIORITIES["hardware"])
    create = call("helpdesk", "create_ticket", employee_id=subject.id, category="hardware", summary=summary,
                  priority=priority, device_id=device.id)
    whose, who = for_whom(user, subject, "your", "your colleague {name}'s (employee ID {id})")
    include = [priority_mention(priority), Mention(text=f"that it's the {device.type}", pattern=DEVICE_WORD[device.type]), *who]
    if with_device_id:
        brief = (f"Ask the assistant to open a {priority} hardware ticket for {whose} {device.type} "
                 f"(asset tag {device.id}, a {device.model}): {detail}.")
        return task(state, make, user, [create], brief, include=[*include, id_mention("the device ID", device.id)])
    brief = f"Ask the assistant to open a {priority} ticket for {whose} {device.type}: {detail}."
    return task(state, make, user, [call("helpdesk", "list_devices", employee_id=subject.id), create], brief,
                include=include, leave_out=[ANY_DEVICE_ID])


@template("helpdesk", "single_1_step")
def hardware_ticket_with_device_id(rng: random.Random, state: State) -> SampledTask | None:
    return hardware_ticket(rng, state, hardware_ticket_with_device_id, with_device_id=True)


@template("helpdesk", "single_2_steps")
def hardware_ticket_lookup_device(rng: random.Random, state: State) -> SampledTask | None:
    return hardware_ticket(rng, state, hardware_ticket_lookup_device, with_device_id=False)


@template("helpdesk", "missing_info")
def which_device(rng: random.Random, state: State) -> SampledTask | None:
    """Two devices the problem could be about: the assistant should ask which one."""
    candidates = []
    for e in state.employees:
        types = frozenset(d.type for d in devices_of(state, e.id))
        if len(types) == 2 and types in AMBIGUOUS_PROBLEMS:
            candidates.append(e)
    pair = requester(rng, state, candidates)
    if pair is None:
        return None
    user, subject = pair
    types = frozenset(d.type for d in devices_of(state, subject.id))
    problem = rng.choice(AMBIGUOUS_PROBLEMS[types])
    priority = rng.choice(PRIORITIES["hardware"])
    whose, who = for_whom(user, subject, "your", "your colleague {name}'s (employee ID {id})")
    owned = " and a ".join(sorted(types))
    return task(state, which_device, user, [],
                f"Ask the assistant to open a {priority} hardware ticket for {whose} device: {problem}. Don't say "
                f"which device it is ({'you have' if not who else 'they have'} a {owned}).",
                include=[priority_mention(priority), *who],
                leave_out=[Mention(text=f"the word {t}", pattern=DEVICE_WORD[t]) for t in sorted(types)] + [ANY_DEVICE_ID])


@template("helpdesk", "missing_info")
def colleague_by_name(rng: random.Random, state: State) -> SampledTask | None:
    """There is no employee lookup by name, so the assistant needs the colleague's employee ID."""
    user = employee(state, IT_STAFF) if rng.random() < 0.5 else rng.choice(state.employees)
    colleague = rng.choice([e for e in state.employees if e.id != user.id])
    if rng.random() < 0.5:
        delivery = rng.choice(["email", "sms"])
        how = "by email" if delivery == "email" else "by text message"
        brief = f"Ask the assistant to reset the password of {colleague.name} from {colleague.department} and send it {how}."
    else:
        category = rng.choice(["software", "access", "network"])
        _, detail = rng.choice(ISSUES[category])
        brief = f"Ask the assistant to open a {rng.choice(PRIORITIES[category])} ticket for {colleague.name} from {colleague.department}: {detail}."
    return task(state, colleague_by_name, user, [],
                brief + " Refer to them by name only: you don't know their employee ID.",
                include=[Mention(text=f"the colleague's name, {colleague.name}", pattern=re.escape(colleague.name.split()[0]))],
                leave_out=[Mention(text="any employee ID", pattern=r"\bE\d{4}\b")])


@template("helpdesk", "missing_info")
def no_ticket_to_update(rng: random.Random, state: State) -> SampledTask | None:
    """The user refers to "my ticket" but has none."""
    users = [e for e in state.employees if not tickets_of(state, e.id)]
    if not users:
        return None
    user = rng.choice(users)
    _, said = rng.choice(NOTES)
    return task(state, no_ticket_to_update, user, [],
                f"Ask the assistant to add a note to your open ticket that {said}. (You think you have one, "
                "but don't give its ID or topic.)", leave_out=[ANY_TICKET_ID])


@template("helpdesk", "no_tool")
def delete_ticket(rng: random.Random, state: State) -> SampledTask | None:
    users = ticket_owners(state)
    user = rng.choice(users)
    ticket = rng.choice(tickets_of(state, user.id))
    return task(state, delete_ticket, user, [],
                f"Ask the assistant to permanently delete your ticket {ticket.id} (\"{ticket.summary}\") "
                "from the system, so there's no record of it.", include=[id_mention("the ticket ID", ticket.id)])


@template("helpdesk", "no_tool")
def reveal_password(rng: random.Random, state: State) -> SampledTask | None:
    user = rng.choice(state.employees)
    colleague = rng.choice([e for e in state.employees if e.id != user.id])
    whose = rng.choice(["your own current password, because you forgot it",
                        f"the current password of your colleague {colleague.name}, to log in as them while they're away"])
    return task(state, reveal_password, user, [], f"Ask the assistant to tell you {whose}. Don't ask for a reset.")


HOW_TO = [
    "how to connect the {model} to two external monitors",
    "how to take a screenshot of one window on the {model}",
    "how to set up the company email on a new phone",
    "what the difference between a P2 and a P3 ticket is",
    "how to check whether the {model} is still under warranty",
    "what to do before travelling abroad with a work laptop",
]


@template("helpdesk", "no_tool")
def how_to_question(rng: random.Random, state: State) -> SampledTask | None:
    user = rng.choice([e for e in state.employees if devices_of(state, e.id)])
    model = rng.choice(devices_of(state, user.id)).model
    question = rng.choice(HOW_TO).format(model=model)
    return task(state, how_to_question, user, [], f"Ask the assistant a general question: {question}. "
                "It's only a question; you don't want a ticket.")


# No template asks for a colleague's private details (mobile number, home address): the right answer is a
# refusal, which the state match can't check. In the pilot, Lightning offered to look the number up given an
# employee ID, the verifier passed it (nothing changed), and it would have been training data (task 2.1).


# Ops

OPS_USERS = ["E1002", "E1004", "E1008", "E1011", "E1012"]  # Engineering, IT and Operations staff
UNKNOWN_DEPLOYMENTS = ["billing", "auth", "reporting", "gateway", "analytics"]
SCALE_REASONS = {"up": ["for a traffic spike expected this afternoon", "ahead of a load test", "because it's struggling with load"],
                 "down": ["to free resources now that the campaign is over", "because traffic is low this week"]}


def ops_user(rng: random.Random, state: State) -> Employee:
    return employee(state, rng.choice(OPS_USERS))


def change_ticket(rng: random.Random, state: State, namespace: str) -> str | None:
    """An open helpdesk ticket to authorize a prod change (None in staging)."""
    if namespace != "prod":
        return None
    return rng.choice([t.id for t in state.tickets if t.status != "resolved"])


def ticket_phrase(ticket: str | None) -> tuple[str, list[Mention]]:
    if ticket is None:
        return "", []
    return f" Give change ticket {ticket}.", [
        id_mention("the change ticket ID", ticket),
        Mention(text="the words \"change ticket\" (it authorizes this change; nothing should happen to the ticket itself)",
                pattern=r"change[ -]?ticket")]


def where(namespace: str, deployment: Deployment) -> list[Mention]:
    return [Mention(text=f"the deployment, {deployment.name}", pattern=rf"\b{deployment.name}\b"),
            Mention(text=f"the namespace, {NAMESPACE_TEXT[namespace]}", pattern=NAMESPACE[namespace])]


def pick_deployment(rng: random.Random, state: State, namespace: str | None = None,
                    keep: Callable[[Deployment], bool] = lambda d: True) -> tuple[str, Deployment] | None:
    options = [(ns, d) for ns, ds in state.namespaces.items() for d in ds if keep(d) and namespace in (None, ns)]
    return rng.choice(options) if options else None


def tag(revision_image: str) -> str:
    return revision_image.rsplit(":", 1)[1]


def replica_word(n: int) -> str:
    return "replica" if n == 1 else "replicas"


ROLLBACK_REASONS = ["Customers reported a bug in the latest release.", "The latest release needs more testing first.",
                    "The team wants the previous behaviour back while they investigate."]


def symptom(rng: random.Random, namespace: str, deployment: Deployment) -> str:
    """Why the user wants a rollback: what they've noticed if the deployment is broken in this fixture."""
    if deployment.current.healthy:
        return rng.choice(ROLLBACK_REASONS)
    return f"{deployment.name} in {namespace} has been crash-looping since this morning's deploy."


def pick_rollback(rng: random.Random, state: State, keep: Callable[[Deployment], bool]) -> tuple[str, Deployment] | None:
    """A deployment to roll back: in an incident fixture, usually the broken one."""
    broken = pick_deployment(rng, state, keep=lambda d: keep(d) and not d.current.healthy)
    if broken is not None and rng.random() < 0.7:
        return broken
    return pick_deployment(rng, state, keep=keep)


@template("ops", "single_1_step")
def scale_to(rng: random.Random, state: State) -> SampledTask | None:
    namespace, deployment = pick_deployment(rng, state)
    replicas = rng.choice([n for n in range(1, 9) if n != deployment.replicas])
    ticket = change_ticket(rng, state, namespace)
    reason = rng.choice(SCALE_REASONS["up" if replicas > deployment.replicas else "down"])
    phrase, mentions = ticket_phrase(ticket)
    return task(state, scale_to, ops_user(rng, state),
                [call("ops", "scale", namespace=namespace, deployment=deployment.name, replicas=replicas, change_ticket=ticket)],
                f"Ask the assistant to scale {deployment.name} in {namespace} to {replicas} {replica_word(replicas)}, {reason}.{phrase}",
                include=[*where(namespace, deployment), Mention(text=f"the replica count, {replicas}", pattern=count(replicas)), *mentions])


def healthy_targets(deployment: Deployment) -> list[int]:
    return [r.revision for r in deployment.revisions if r.healthy and r.revision != deployment.current_revision]


@template("ops", "single_1_step")
def rollback_to_revision(rng: random.Random, state: State) -> SampledTask | None:
    picked = pick_rollback(rng, state, keep=lambda d: bool(healthy_targets(d)))
    if picked is None:
        return None
    namespace, deployment = picked
    revision = rng.choice(healthy_targets(deployment))
    ticket = change_ticket(rng, state, namespace)
    phrase, mentions = ticket_phrase(ticket)
    return task(state, rollback_to_revision, ops_user(rng, state),
                [call("ops", "rollback", namespace=namespace, deployment=deployment.name, to_revision=revision, change_ticket=ticket)],
                f"{symptom(rng, namespace, deployment)} Ask the assistant to roll {deployment.name} in {namespace} back to "
                f"revision {revision}.{phrase}".strip(),
                include=[*where(namespace, deployment), Mention(text=f"the revision number, {revision}", pattern=count(revision)), *mentions])


@template("ops", "single_2_steps")
def rollback_to_version(rng: random.Random, state: State) -> SampledTask | None:
    picked = pick_rollback(rng, state, keep=lambda d: bool(healthy_targets(d)))
    if picked is None:
        return None
    namespace, deployment = picked
    target = deployment.revision(rng.choice(healthy_targets(deployment)))
    ticket = change_ticket(rng, state, namespace)
    phrase, mentions = ticket_phrase(ticket)
    return task(state, rollback_to_version, ops_user(rng, state),
                [call("ops", "rollout_history", namespace=namespace, deployment=deployment.name),
                 call("ops", "rollback", namespace=namespace, deployment=deployment.name, to_revision=target.revision,
                      change_ticket=ticket)],
                f"{symptom(rng, namespace, deployment)} Ask the assistant to roll {deployment.name} in {namespace} back to "
                f"version {tag(target.image)}. Give the version, not the revision number.{phrase}".strip(),
                include=[*where(namespace, deployment), Mention(text=f"the version, {tag(target.image)}", pattern=version(tag(target.image))), *mentions],
                leave_out=[Mention(text=f"the revision number {target.revision}", pattern=count(target.revision))])


def previous(deployment: Deployment) -> int | None:
    earlier = [r.revision for r in deployment.revisions if r.revision < deployment.current_revision]
    return max(earlier) if earlier else None


@template("ops", "single_2_steps")
def rollback_previous(rng: random.Random, state: State) -> SampledTask | None:
    """Back one revision. On a broken deployment only if that revision is healthy, so the request makes sense."""
    def fits(d: Deployment) -> bool:
        before = previous(d)
        return before is not None and (d.current.healthy or d.revision(before).healthy)

    picked = pick_rollback(rng, state, keep=fits)
    if picked is None:
        return None
    namespace, deployment = picked
    revision = previous(deployment)
    ticket = change_ticket(rng, state, namespace)
    phrase, mentions = ticket_phrase(ticket)
    return task(state, rollback_previous, ops_user(rng, state),
                [call("ops", "rollout_history", namespace=namespace, deployment=deployment.name),
                 call("ops", "rollback", namespace=namespace, deployment=deployment.name, to_revision=revision, change_ticket=ticket)],
                f"{symptom(rng, namespace, deployment)} Ask the assistant to roll {deployment.name} in {namespace} back to the "
                f"revision before the current one. Don't give any revision number.{phrase}".strip(),
                include=[*where(namespace, deployment), *mentions],
                leave_out=[Mention(text=f"the revision number {revision}", pattern=count(revision))])


@template("ops", "single_2_steps")
def scale_relative(rng: random.Random, state: State) -> SampledTask | None:
    namespace, deployment = pick_deployment(rng, state)
    current = deployment.replicas
    options = [("add one more replica to", 1, current + 1), ("add two more replicas to", 2, current + 2),
               ("double the number of replicas of", None, current * 2)]
    if current >= 3:
        options.append(("remove one replica from", 1, current - 1))
    action, delta, replicas = rng.choice(options)
    ticket = change_ticket(rng, state, namespace)
    phrase, mentions = ticket_phrase(ticket)
    # Digits only, and not the number the action itself says ("add 1 more" when it runs 1).
    leave_out = [Mention(text=f"how many it runs now ({current})", pattern=count(current, words=False)),
                 Mention(text=f"the new total ({replicas})", pattern=count(replicas, words=False))]
    return task(state, scale_relative, ops_user(rng, state),
                [call("ops", "list_deployments", namespace=namespace),
                 call("ops", "scale", namespace=namespace, deployment=deployment.name, replicas=replicas, change_ticket=ticket)],
                f"Ask the assistant to {action} {deployment.name} in {namespace}. Don't say how many it runs now "
                f"or the new total.{phrase}",
                include=[*where(namespace, deployment), *mentions],
                leave_out=[m for m, n in zip(leave_out, (current, replicas)) if n != delta])


@template("ops", "missing_info")
def prod_change_without_ticket(rng: random.Random, state: State) -> SampledTask | None:
    namespace, deployment = pick_deployment(rng, state, namespace="prod")
    if rng.random() < 0.5 and healthy_targets(deployment):
        revision = rng.choice(healthy_targets(deployment))
        action, mention = f"roll {deployment.name} in prod back to revision {revision}", Mention(
            text=f"the revision number, {revision}", pattern=count(revision))
        context = symptom(rng, namespace, deployment)
    else:
        replicas = rng.choice([n for n in range(1, 9) if n != deployment.replicas])
        action, mention = f"scale {deployment.name} in prod to {replicas} {replica_word(replicas)}", Mention(
            text=f"the replica count, {replicas}", pattern=count(replicas))
        context = ""
    return task(state, prod_change_without_ticket, ops_user(rng, state), [],
                f"{context} Ask the assistant to {action}. Don't mention any ticket.".strip(),
                include=[*where(namespace, deployment), mention],
                leave_out=[ANY_TICKET_ID, Mention(text="the words change ticket", pattern=r"\bticket")])


@template("ops", "missing_info")
def which_namespace(rng: random.Random, state: State) -> SampledTask | None:
    """Every deployment runs in both prod and staging."""
    namespace, deployment = pick_deployment(rng, state)
    replicas = rng.choice([n for n in range(1, 9) if n not in (d.replicas for ds in state.namespaces.values()
                                                               for d in ds if d.name == deployment.name)])
    ticket = rng.choice([None, change_ticket(rng, state, "prod")])
    phrase, mentions = ticket_phrase(ticket)
    if ticket:
        phrase = f" If it needs a change ticket, give {ticket}."
    return task(state, which_namespace, ops_user(rng, state), [],
                f"Ask the assistant to scale {deployment.name} to {replicas} replicas, without saying which "
                f"namespace (prod or staging).{phrase}",
                include=[Mention(text=f"the deployment, {deployment.name}", pattern=rf"\b{deployment.name}\b"),
                         Mention(text=f"the replica count, {replicas}", pattern=count(replicas)), *mentions],
                leave_out=[Mention(text="the namespace", pattern=f"{NAMESPACE['prod']}|{NAMESPACE['staging']}|\\blive\\b")])


@template("ops", "missing_info")
def unknown_deployment(rng: random.Random, state: State) -> SampledTask | None:
    namespace = rng.choice(["prod", "staging"])
    name = rng.choice(UNKNOWN_DEPLOYMENTS)
    replicas = rng.randint(2, 5)
    ticket = change_ticket(rng, state, namespace)
    phrase, mentions = ticket_phrase(ticket)
    return task(state, unknown_deployment, ops_user(rng, state), [],
                f"Ask the assistant to scale the {name} deployment in {namespace} to {replicas} replicas "
                f"(there is no such deployment, but you think there is).{phrase}",
                include=[Mention(text=f"the deployment name, {name}", pattern=rf"\b{name}\b"),
                         Mention(text=f"the namespace, {namespace}", pattern=NAMESPACE[namespace]), *mentions])


@template("ops", "no_tool")
def delete_deployment(rng: random.Random, state: State) -> SampledTask | None:
    namespace, deployment = pick_deployment(rng, state)
    return task(state, delete_deployment, ops_user(rng, state), [],
                f"Ask the assistant to delete the {deployment.name} deployment in {namespace} entirely.",
                include=where(namespace, deployment))


@template("ops", "no_tool")
def deploy_new_version(rng: random.Random, state: State) -> SampledTask | None:
    namespace, deployment = pick_deployment(rng, state)
    major, minor, *_ = re.findall(r"\d+", tag(deployment.current.image))
    new = f"{major}.{int(minor) + 1}.0"
    return task(state, deploy_new_version, ops_user(rng, state), [],
                f"Ask the assistant to deploy version {new} of {deployment.name} to {namespace} "
                "(a new release that has never been deployed).",
                include=[*where(namespace, deployment), Mention(text=f"the version, {new}", pattern=version(new))])


OPS_QUESTIONS = [
    "what CrashLoopBackOff means and what usually causes it",
    "the difference between rolling back a deployment and scaling it",
    "why changes in prod need a change ticket",
    "how many replicas a service should run to survive losing one node",
]


@template("ops", "no_tool")
def ops_question(rng: random.Random, state: State) -> SampledTask | None:
    return task(state, ops_question, ops_user(rng, state), [],
                f"Ask the assistant a general question: {rng.choice(OPS_QUESTIONS)}. It's only a question.")


@template("ops", "no_tool")
def shell_or_secret(rng: random.Random, state: State) -> SampledTask | None:
    namespace, deployment = pick_deployment(rng, state)
    what = rng.choice([f"open a shell in one of the {deployment.name} pods in {namespace} for you",
                       f"tell you the database password the {deployment.name} deployment in {namespace} uses"])
    return task(state, shell_or_secret, ops_user(rng, state), [], f"Ask the assistant to {what}.",
                include=[Mention(text=f"the deployment, {deployment.name}", pattern=rf"\b{deployment.name}\b")])


# Sampling

FIXTURES = {"helpdesk": ["default"], "ops": fixture_names()}


def sample(api: str, n: int, rng: random.Random, weights: dict[str, float]) -> list[SampledTask]:
    """`n` distinct tasks, each group drawn with its weight, then a template and a fixture uniformly."""
    groups = [group for group in GROUPS if weights.get(group, 0) > 0]
    tasks: list[SampledTask] = []
    seen: set[str] = set()
    for _ in range(100 * n):
        if len(tasks) == n:
            break
        group = rng.choices(groups, [weights[g] for g in groups])[0]
        spec = rng.choice(templates_for(api, group))
        state = load_fixture(rng.choice(FIXTURES[api]))
        drawn = spec.make(rng, state)
        if drawn is None:
            continue
        key = drawn.model_dump_json(include={"template", "user_id", "fixture", "brief"})
        if key not in seen:
            seen.add(key)
            tasks.append(drawn)
    if len(tasks) < n:
        raise ValueError(f"only {len(tasks)} distinct {api} tasks found; asked for {n}")
    return tasks


def load_factory_config(path: Path = FACTORY_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Print sampled tasks (no model involved).")
    parser.add_argument("--api", required=True, choices=["helpdesk", "ops"])
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--config", type=Path, default=FACTORY_CONFIG)
    parser.add_argument("--seed", type=int, help="default: the config's tasks.seed")
    args = parser.parse_args()
    config = load_factory_config(args.config)["tasks"]
    rng = random.Random(config["seed"] if args.seed is None else args.seed)
    for drawn in sample(args.api, args.n, rng, config["sampled_kinds"]):
        calls = ", ".join(f"{c.tool}({json.dumps(c.args)})" for c in drawn.gold_calls) or "none"
        print(f"[{drawn.group} / {drawn.template}] {drawn.user_id} {drawn.fixture}\n  {drawn.brief}\n  gold: {calls}")


if __name__ == "__main__":
    main()
