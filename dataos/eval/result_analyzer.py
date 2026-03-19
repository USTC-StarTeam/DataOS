"""Analysis helpers that summarize evaluation outputs for controllers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dataos.core.eval import EvalResult
from dataos.core.optimization import OptimizationState
from dataos.core.task import TaskSpec

__all__ = [
    "AnalysisSummary",
    "ResultAnalyzer",
]


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(value)


@dataclass(slots=True)
class AnalysisSummary:
    """Condensed cross-result analysis consumed by a controller."""

    summary_id: str
    winner_eval_id: str | None = None
    winner_dataset_version: str | None = None
    metric_deltas: dict[str, float] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.summary_id:
            raise ValueError("summary_id must not be empty")
        self.metric_deltas = dict(self.metric_deltas)
        self.observations = list(self.observations)
        self.metadata = _copy_mapping(self.metadata)


@runtime_checkable
class ResultAnalyzer(Protocol):
    """Boundary contract for turning raw eval results into controller-ready signals."""

    def summarize(
        self,
        task_spec: TaskSpec,
        results: list[EvalResult],
        optimization_state: OptimizationState | None = None,
    ) -> AnalysisSummary:
        """Summarize candidate evaluation results without mutating runtime state."""
