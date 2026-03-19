"""Retry interfaces shared by runtime execution components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dataos.core.pipeline import PipelineStep

__all__ = [
    "RetryDecision",
    "RetryPolicy",
]


@dataclass(slots=True)
class RetryDecision:
    """Decision emitted by a retry policy for one failed step attempt."""

    should_retry: bool
    delay_seconds: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        self.metadata = dict(self.metadata)


@runtime_checkable
class RetryPolicy(Protocol):
    """Boundary contract for step-level retry decisions inside runtime."""

    def decide(self, step: PipelineStep, error: Exception, attempt: int) -> RetryDecision:
        """Return whether runtime should retry a failed step attempt."""
