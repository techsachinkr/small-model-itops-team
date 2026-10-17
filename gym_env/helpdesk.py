"""Helpdesk tools: devices, tickets and password resets."""

from typing import Annotated, Any

from pydantic import Field

from gym_env.state import (
    Category,
    Delivery,
    Device,
    Employee,
    PasswordReset,
    Priority,
    State,
    Ticket,
    TicketStatus,
)
from gym_env.tools import Args, Tool, ToolError

EmployeeId = Annotated[str, Field(pattern=r"^E\d{4}$", description="Employee ID, like E1001.")]
TicketId = Annotated[str, Field(pattern=r"^HD-\d+$", description="Ticket ID, like HD-1042.")]


def find_employee(state: State, employee_id: str) -> Employee:
    for employee in state.employees:
        if employee.id == employee_id:
            return employee
    raise ToolError("not_found", f"Employee {employee_id} not found.")


def find_ticket(state: State, ticket_id: str) -> Ticket:
    for ticket in state.tickets:
        if ticket.id == ticket_id:
            return ticket
    raise ToolError(
        "not_found",
        f"Ticket {ticket_id} not found. Use list_tickets(employee_id=...) to find ticket IDs.",
    )


def _find_device(state: State, device_id: str) -> Device:
    for device in state.devices:
        if device.id == device_id:
            return device
    raise ToolError(
        "not_found",
        f"Device {device_id} not found. Use list_devices(employee_id=...) to find device IDs.",
    )


def _ticket_view(ticket: Ticket) -> dict[str, Any]:
    return ticket.model_dump(exclude_none=True)


class ListDevicesArgs(Args):
    employee_id: EmployeeId


def list_devices(state: State, args: ListDevicesArgs) -> dict[str, Any]:
    """List the devices assigned to an employee. Hardware tickets need a device_id from this list."""
    find_employee(state, args.employee_id)
    devices = [d.model_dump(exclude={"employee_id"}) for d in state.devices if d.employee_id == args.employee_id]
    return {"employee_id": args.employee_id, "devices": devices}


class CreateTicketArgs(Args):
    employee_id: str = Field(
        pattern=r"^E\d{4}$", description="The employee the ticket is for, like E1001."
    )
    category: Category = Field(
        description=(
            "hardware: a problem with a physical device (needs device_id). "
            "software: an application problem or software request on a user's machine. "
            "access: accounts, logins and permissions. "
            "network: VPN, Wi-Fi and connectivity. "
            "incident: an outage or failure of a shared or production service."
        )
    )
    summary: str = Field(min_length=1, description="One line describing the problem.")
    priority: Priority = Field(description="P1 is the most urgent, P4 the least.")
    device_id: str | None = Field(
        None,
        pattern=r"^D\d{4}$",
        description="The affected device's ID from list_devices. Required for hardware tickets.",
    )


def create_ticket(state: State, args: CreateTicketArgs) -> dict[str, Any]:
    """Open a helpdesk ticket for an employee. Returns the new ticket with its ID.
    Hardware tickets need the device_id of the affected device (see list_devices)."""
    find_employee(state, args.employee_id)
    if not args.summary.strip():
        raise ToolError("invalid_arguments", "summary must not be blank.")
    if args.category == "hardware" and args.device_id is None:
        raise ToolError(
            "failed_precondition",
            "Hardware tickets need device_id. Call "
            f"list_devices(employee_id='{args.employee_id}') to find the affected device, then retry.",
        )
    if args.device_id is not None:
        device = _find_device(state, args.device_id)
        if device.employee_id != args.employee_id:
            raise ToolError(
                "failed_precondition",
                f"Device {device.id} is assigned to {device.employee_id}, not {args.employee_id}. "
                f"Call list_devices(employee_id='{args.employee_id}') to see this employee's devices.",
            )
    ticket = Ticket(
        id=state.next_id("ticket", "HD"),
        employee_id=args.employee_id,
        category=args.category,
        summary=args.summary.strip(),
        priority=args.priority,
        status="open",
        device_id=args.device_id,
        created_at=state.now,
        updated_at=state.now,
    )
    state.tickets.append(ticket)
    return {"ticket": _ticket_view(ticket)}


class ListTicketsArgs(Args):
    employee_id: EmployeeId
    status: TicketStatus | None = Field(None, description="Only tickets with this status.")


def list_tickets(state: State, args: ListTicketsArgs) -> dict[str, Any]:
    """List an employee's tickets, newest first, optionally filtered by status."""
    find_employee(state, args.employee_id)
    tickets = [
        t.model_dump(include={"id", "category", "summary", "priority", "status", "updated_at"})
        for t in reversed(state.tickets)
        if t.employee_id == args.employee_id and args.status in (None, t.status)
    ]
    return {"employee_id": args.employee_id, "tickets": tickets}


class GetTicketArgs(Args):
    ticket_id: TicketId


def get_ticket(state: State, args: GetTicketArgs) -> dict[str, Any]:
    """Get one ticket with all its details and notes."""
    return {"ticket": _ticket_view(find_ticket(state, args.ticket_id))}


class UpdateTicketArgs(Args):
    ticket_id: TicketId
    status: TicketStatus | None = None
    priority: Priority | None = None
    note: str | None = Field(None, min_length=1, description="A note to add to the ticket.")


def update_ticket(state: State, args: UpdateTicketArgs) -> dict[str, Any]:
    """Change a ticket's status or priority, and/or add a note. Give at least one of them."""
    if args.status is None and args.priority is None and args.note is None:
        raise ToolError(
            "invalid_arguments", "Nothing to update: give at least one of status, priority or note."
        )
    if args.note is not None and not args.note.strip():
        raise ToolError("invalid_arguments", "note must not be blank.")
    ticket = find_ticket(state, args.ticket_id)
    changed = False
    for field in ("status", "priority"):
        value = getattr(args, field)
        if value is not None and value != getattr(ticket, field):
            setattr(ticket, field, value)
            changed = True
    if args.note is not None:
        ticket.notes.append(args.note.strip())
        changed = True
    if changed:
        ticket.updated_at = state.now
    return {"ticket": _ticket_view(ticket), "changed": changed}


class ResetPasswordArgs(Args):
    employee_id: EmployeeId
    delivery: Delivery = Field(description="How to send the temporary password.")


def reset_password(state: State, args: ResetPasswordArgs) -> dict[str, Any]:
    """Reset an employee's password and send them a temporary one by email or SMS."""
    employee = find_employee(state, args.employee_id)
    state.password_resets.append(
        PasswordReset(employee_id=employee.id, delivery=args.delivery, at=state.now)
    )
    sent_to = employee.email if args.delivery == "email" else "the mobile number on file"
    return {"employee_id": employee.id, "delivery": args.delivery, "sent_to": sent_to}


TOOLS = [
    Tool("list_devices", "helpdesk", ListDevicesArgs, list_devices),
    Tool("create_ticket", "helpdesk", CreateTicketArgs, create_ticket),
    Tool("list_tickets", "helpdesk", ListTicketsArgs, list_tickets),
    Tool("get_ticket", "helpdesk", GetTicketArgs, get_ticket),
    Tool("update_ticket", "helpdesk", UpdateTicketArgs, update_ticket),
    Tool("reset_password", "helpdesk", ResetPasswordArgs, reset_password),
]
