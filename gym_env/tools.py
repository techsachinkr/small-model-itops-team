"""Tool primitives shared by helpdesk and ops: argument models, readable errors and generated schemas."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from gym_env.state import State

ApiName = Literal["helpdesk", "ops"]


class Args(BaseModel):
    """Tool arguments. Unknown arguments and type coercion are rejected, so the env accepts exactly
    what the generated JSON schema allows."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ToolError(Exception):
    """An error returned to the model as the tool result, written so the model can recover."""

    def __init__(self, code: str, message: str, status: int = 400, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, **self.details}}


@dataclass(frozen=True)
class Tool:
    name: str
    api: ApiName
    args: type[Args]
    run: Callable[["State", Any], dict[str, Any]]

    @property
    def description(self) -> str:
        return inspect.getdoc(self.run) or ""

    def signature(self) -> str:
        """Like `create_ticket(employee_id, category, summary, priority, device_id?)`."""
        names = [
            name if field.is_required() else f"{name}?"
            for name, field in self.args.model_fields.items()
        ]
        return f"{self.name}({', '.join(names)})"

    def schema(self) -> dict[str, Any]:
        """The tool as a Responses API function definition, generated from its argument model."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": _clean_schema(self.args.model_json_schema()),
            "strict": False,
        }

    def validate(self, raw: Any) -> Args:
        try:
            return self.args.model_validate(raw)
        except ValidationError as error:
            problems = [_describe(e) for e in error.errors()]
            raise ToolError(
                "invalid_arguments",
                f"Invalid arguments for {self.name}: {'; '.join(problems)}. "
                f"Expected {self.signature()} (? = optional).",
                problems=problems,
            ) from None


def _describe(error: Any) -> str:
    field = ".".join(str(part) for part in error["loc"])
    if error["type"] == "missing":
        return f"missing required argument '{field}'"
    if error["type"] == "extra_forbidden":
        return f"unknown argument '{field}'"
    if not field:
        return f"arguments must be a JSON object (got {error['input']!r})"
    return f"{field}: {error['msg']} (got {error['input']!r})"


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop pydantic's titles and turn `X | None = None` into an optional `X`."""
    properties = {}
    for name, prop in schema.get("properties", {}).items():
        prop = {key: value for key, value in prop.items() if key != "title"}
        if "anyOf" in prop:
            (option,) = [o for o in prop.pop("anyOf") if o.get("type") != "null"]
            prop = {**option, **prop}
            if prop.get("default", ...) is None:
                del prop["default"]
        properties[name] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": schema.get("required", []),
        "additionalProperties": False,
    }
