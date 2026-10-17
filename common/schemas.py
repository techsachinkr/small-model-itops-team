"""Data formats shared across the project (CLAUDE.md "Data formats")."""

from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    model_validator,
)

Api = Literal["helpdesk", "ops", "calendar"]
Kind = Literal["single", "multi", "no_tool", "missing_info", "read_only"]
Setup = Literal[
    "base_raw",
    "base_seatbelt",
    "r1",
    "r2",
    "lightning",
    "single_base",
    "single_trained",
    "single_lightning",
    "team_base",
    "team_trained",
    "team_escalate",
    "team_small_planner",
]


def _check_fraction(value: str) -> str:
    passed, total = (int(part) for part in value.split("/"))
    if passed > total:
        raise ValueError(f"{value}: numerator is larger than denominator")
    return value


# Results are reported as fractions like "37/50", never percentages.
Fraction = Annotated[
    str, StringConstraints(pattern=r"^\d+/\d+$"), AfterValidator(_check_fraction)
]


class Strict(BaseModel):
    """Base model that rejects unknown fields, so typos fail loudly."""

    model_config = ConfigDict(extra="forbid")


class GoldCall(Strict):
    api: Api
    tool: str
    args: dict[str, Any]


class Case(Strict):
    """One request with its expected calls (data/practice, data/test, data/generated)."""

    id: str
    request: str
    user_id: str
    fixture: str
    apis: list[Api] = Field(min_length=1)
    gold_calls: list[GoldCall]
    kind: Kind
    answer_must_contain: list[str] | None = None
    answer_must_not_contain: list[str] | None = None  # e.g. what an over-inclusive answer would list

    @model_validator(mode="after")
    def _consistent(self) -> "Case":
        if any(call.api not in self.apis for call in self.gold_calls):
            raise ValueError("every gold call's api must be listed in apis")
        if self.kind == "single" and len(self.apis) != 1:
            raise ValueError("single cases use exactly one api")
        if self.kind == "multi" and len(self.apis) < 2:
            raise ValueError("multi cases use at least two apis")
        if self.kind in ("single", "multi") and not self.gold_calls:
            raise ValueError(f"{self.kind} cases need gold_calls")
        if self.kind in ("no_tool", "missing_info") and self.gold_calls:
            raise ValueError(f"{self.kind} cases have no gold_calls")
        if self.kind == "read_only" and not self.answer_must_contain:
            raise ValueError("read_only cases need answer_must_contain")
        required = [text.casefold() for text in self.answer_must_contain or []]
        if any(text.casefold() in r for text in self.answer_must_not_contain or [] for r in required):
            raise ValueError("an answer_must_not_contain string is part of a required one")
        return self


class TrainingExample(Strict):
    """One verified Lightning trajectory (data/generated/<api>/round_<n>.jsonl)."""

    id: str
    api: Api
    round: int = Field(ge=1)
    kind: Kind
    origin: Literal["sampled", "written"]
    source: Literal["broad", "targeted"]
    cluster: str | None = None
    case_id: str
    teacher: str
    reward: float
    messages: list[dict[str, Any]]
    text: str

    @model_validator(mode="after")
    def _targeted_has_cluster(self) -> "TrainingExample":
        if self.source == "targeted" and not self.cluster:
            raise ValueError("targeted examples name their failure cluster")
        return self


class Call(Strict):
    """One tool call the model made, and what came back."""

    turn: int = Field(ge=1)
    tool: str
    args: Any  # parsed arguments, or the raw string if they weren't valid JSON
    status: int  # 200 ok, 400/404 tool error, 503 transient; 0 = blocked by the seatbelt, never run
    output: Any


SeatbeltEventKind = Literal[
    "invalid_call",  # unknown tool, unparseable or schema-invalid arguments; not executed
    "tool_error",  # the environment returned an error
    "transient_retry",  # a 503 retried automatically, without the model
    "self_check",  # the completion self-check's verdict
    "gave_up",  # the step failed: retry limit, max steps, or a failed self-check with no retries left
]


class SeatbeltEvent(Strict):
    kind: SeatbeltEventKind
    turn: int = Field(ge=1)
    tool: str | None = None
    detail: str
    complete: bool | None = None  # self_check only
    retries_left: int | None = None  # after this event, for events that use up a retry


class Step(Strict):
    """One step of a trace: one agent working on one part of the request."""

    agent: str
    model: str
    calls: list[Call]
    seatbelt_events: list[SeatbeltEvent]
    escalated: bool = False

    @property
    def gave_up(self) -> bool:
        return any(event.kind == "gave_up" for event in self.seatbelt_events)


class Trace(Strict):
    """One attempt at one case (runs/<run_id>/traces.jsonl)."""

    run_id: str
    setup: Setup
    case_id: str
    attempt: int = Field(ge=1)
    steps: list[Step]
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0)
    cost_usd: float = Field(ge=0)
    passed: bool
    failure_label: Literal["routing", "specialist"] | None = None

    @model_validator(mode="after")
    def _label_only_on_failure(self) -> "Trace":
        if self.passed and self.failure_label is not None:
            raise ValueError("a passed trace has no failure_label")
        return self


class SetupSummary(Strict):
    passes: Fraction
    mean_latency_s: float = Field(ge=0)
    cost_per_1k_usd: float = Field(ge=0)
    small_model_steps: Fraction | None = None  # team_escalate only


class RunSummary(RootModel[dict[Setup, SetupSummary]]):
    """Per-setup results of a run (runs/<run_id>/summary.json)."""


class FailureCluster(Strict):
    name: str
    description: str
    case_ids: list[str] = Field(min_length=1)
    guidance: str


class Clusters(RootModel[list[FailureCluster]]):
    """The failure analyst's output (runs/<run_id>/clusters.json)."""
