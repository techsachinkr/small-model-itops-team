"""Reliability layer: schema validation, retries and the completion self-check (CLAUDE.md "Seatbelt").

The seatbelt makes the decisions and keeps one step's bookkeeping; common/tool_loop.py does the calling.
- check_call(): before a call runs, the tool must exist and its arguments must parse and validate
  against the tool's JSON schema. If not, the error goes back to the model as the tool result.
- tool_error(): environment errors also go back to the model.
- Every invalid call, environment error and failed self-check uses one of the step's retries. Once
  none are left, the next one fails the step (`exhausted`).
- transient_delays(): a 503 is retried automatically with backoff, without involving the model.
- self_check(): when the model gives its final answer, it is asked for a structured
  {complete, reason} verdict. If it isn't complete and retries remain, the step continues.
Every decision is logged as a SeatbeltEvent for the trace.
"""

import json
from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from common.eval_config import SeatbeltConfig
from common.schemas import SeatbeltEvent, SeatbeltEventKind

SELF_CHECK_PROMPT = (
    "Check your reply before it goes to the user. Given the user's request and the tool calls made so "
    "far, is your task complete? It is complete when one of these is true: (a) every action the request "
    "asked for has been carried out; (b) the request is missing information you need, and your reply "
    "asks the user for it; (c) the request should not be carried out, and your reply declines it. It is "
    "not complete only if there is still a tool call you should make now. In `reason`, say which of "
    "(a), (b) or (c) applies, or what is left to do; then set `complete`. Answer with the JSON object only."
)
SELF_CHECK_RETRY = "Your self-check says the task is not complete: {reason} Continue the task."


class SelfCheck(BaseModel):
    complete: bool
    reason: str


# `reason` comes first so the model states which case applies before it decides (the smoke run in
# task 1.2 showed correct requests for missing information judged incomplete).
SELF_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "complete": {"type": "boolean"}},
    "required": ["reason", "complete"],
    "additionalProperties": False,
}


def error_output(code: str, message: str) -> dict[str, Any]:
    """Same shape as the environment's errors (gym_env/tools.py), so the model sees one format."""
    return {"error": {"code": code, "message": message}}


def _describe(error: ValidationError) -> str:
    field = ".".join(str(part) for part in error.absolute_path)
    return f"{field}: {error.message}" if field else error.message


class Seatbelt:
    """One step's seatbelt. `tools` are OpenAI chat-format tool definitions."""

    def __init__(self, config: SeatbeltConfig, tools: list[dict[str, Any]]) -> None:
        self.config = config
        self.validators = {
            tool["function"]["name"]: Draft202012Validator(tool["function"].get("parameters") or {})
            for tool in tools
        }
        self.failures = 0
        self.events: list[SeatbeltEvent] = []

    @property
    def retries_left(self) -> int:
        return max(self.config.max_retries_per_step - self.failures, 0)

    @property
    def exhausted(self) -> bool:
        """A failure happened with no retries left: the step has failed."""
        return self.failures > self.config.max_retries_per_step

    def _log(self, kind: SeatbeltEventKind, turn: int, detail: str, **fields: Any) -> None:
        self.events.append(SeatbeltEvent(kind=kind, turn=turn, detail=detail, **fields))

    def _fail(self, kind: SeatbeltEventKind, turn: int, detail: str, **fields: Any) -> None:
        self.failures += 1
        self._log(kind, turn, detail, retries_left=self.retries_left, **fields)

    def check_call(self, turn: int, tool: str, raw_arguments: str) -> tuple[Any, dict[str, Any] | None]:
        """The parsed arguments, and the error to send back instead of running the call, if any."""
        validator = self.validators.get(tool)
        if validator is None:
            message = f"Unknown tool {tool!r}. Available tools: {', '.join(self.validators)}."
            return raw_arguments, self._invalid(turn, tool, "unknown_tool", message)
        try:
            args = json.loads(raw_arguments) if raw_arguments.strip() else {}
        except json.JSONDecodeError as error:
            message = f"Arguments for {tool} are not valid JSON ({error.msg} at position {error.pos})."
            return raw_arguments, self._invalid(turn, tool, "invalid_json", message)
        problems = [_describe(e) for e in sorted(validator.iter_errors(args), key=lambda e: list(e.path))]
        if problems:
            message = f"Invalid arguments for {tool}: {'; '.join(problems)}."
            return args, self._invalid(turn, tool, "invalid_arguments", message)
        return args, None

    def _invalid(self, turn: int, tool: str, code: str, message: str) -> dict[str, Any]:
        self._fail("invalid_call", turn, message, tool=tool)
        return error_output(code, message)

    def tool_error(self, turn: int, tool: str, status: int, output: Any) -> None:
        try:
            message = output["error"]["message"]
        except (KeyError, TypeError):
            message = json.dumps(output)[:500]
        self._fail("tool_error", turn, f"{status}: {message}", tool=tool)

    def transient_delays(self) -> list[float]:
        """Seconds to wait before each automatic retry of a 503."""
        return [self.config.transient_backoff_s * 2**i for i in range(self.config.transient_retries)]

    def transient_retry(self, turn: int, tool: str, attempt: int, delay_s: float) -> None:
        self._log("transient_retry", turn, f"503; automatic retry {attempt} after {delay_s:g} s", tool=tool)

    def self_check(self, turn: int, answer: str) -> SelfCheck:
        """Parse and log the self-check's answer. A verdict that isn't complete uses a retry."""
        try:
            verdict = SelfCheck.model_validate_json(answer)
        except PydanticValidationError:
            verdict = SelfCheck(complete=False, reason=f"The self-check answer was not valid: {answer[:200]!r}.")
        if verdict.complete:
            self._log("self_check", turn, verdict.reason, complete=True)
        else:
            self._fail("self_check", turn, verdict.reason, complete=False)
        return verdict

    def give_up(self, turn: int, reason: str) -> None:
        self._log("gave_up", turn, reason)
