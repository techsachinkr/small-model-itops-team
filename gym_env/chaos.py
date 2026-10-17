"""Seeded transient 503 errors, off by default."""

import random

from pydantic import BaseModel, Field

from gym_env.tools import ToolError


class ChaosConfig(BaseModel):
    """On only in the demo or in runs flagged --chaos (CLAUDE.md "Environment rules")."""

    enabled: bool = False
    rate: float = Field(0.15, ge=0, le=1)  # fraction of tool calls that fail
    seed: int = 0


class Chaos:
    """Decides which calls of one task fail. One draw per call, so the pattern depends only on
    the seed, the task's key (its case ID) and the call's position."""

    def __init__(self, config: ChaosConfig, key: str) -> None:
        self.config = config
        self._rng = random.Random(f"{config.seed}:{key}")

    def strike(self) -> bool:
        return self.config.enabled and self._rng.random() < self.config.rate


def unavailable(api: str) -> ToolError:
    return ToolError(
        "service_unavailable",
        f"503 Service Unavailable: the {api} service is temporarily unavailable. "
        "Nothing was changed; retry the same call.",
        status=503,
        retryable=True,
    )
