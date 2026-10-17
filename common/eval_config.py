"""Shared eval settings from configs/eval.yaml (hard rule 7: the same for every setup)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO = Path(__file__).resolve().parent.parent
EVAL_CONFIG = REPO / "configs" / "eval.yaml"


class SeatbeltConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_retries_per_step: int = Field(ge=0)
    transient_retries: int = Field(ge=0)
    transient_backoff_s: float = Field(ge=0)


class Prices(BaseModel):
    prompt: float = Field(ge=0)  # USD per 1M tokens
    completion: float = Field(ge=0)


class EvalConfig(BaseModel):
    temperature: float = Field(ge=0)
    top_p: float = Field(gt=0, le=1)
    max_tokens: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    attempts_per_case: int = Field(gt=0)
    pass_min_attempts: int = Field(gt=0)
    seatbelt: SeatbeltConfig
    chaos: bool
    prices_usd_per_1m_tokens: dict[str, Prices]

    def sampling(self) -> dict[str, float | int]:
        return {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens}

    def prices_for(self, model: str) -> Prices:
        """Adapters are priced as the student (configs/eval.yaml)."""
        return self.prices_usd_per_1m_tokens.get(model, self.prices_usd_per_1m_tokens["student"])


def load_eval_config(path: Path = EVAL_CONFIG) -> EvalConfig:
    return EvalConfig.model_validate(yaml.safe_load(path.read_text()))
