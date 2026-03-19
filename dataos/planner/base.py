"""Planning interfaces and handoff objects for DataOS."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dataos.core.dataset import DatasetVersion
from dataos.core.optimization import OptimizationState
from dataos.core.pipeline import PipelineStep
from dataos.core.task import TaskSpec
from dataos.registry.operator_registry import OperatorRegistry

__all__ = [
    "Planner",
    "PlannerRequest",
    "PlanningResult",
]


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(value)


@dataclass(slots=True)
class PlannerRequest:
    """Stable planner input assembled by a controller."""

    task_spec: TaskSpec
    datasets: list[DatasetVersion] = field(default_factory=list)
    optimization_state: OptimizationState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.datasets = list(self.datasets)
        self.metadata = _copy_mapping(self.metadata)


@dataclass(slots=True)
class PlanningResult:
    """Planner output whose pipeline is runtime-only and excludes evaluation execution."""

    plan_id: str
    planner_id: str
    pipeline: list[PipelineStep] = field(default_factory=list)
    candidate_config: dict[str, Any] = field(default_factory=dict)
    rationale: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.plan_id, "plan_id")
        _require_text(self.planner_id, "planner_id")
        self.pipeline = list(self.pipeline)
        self.candidate_config = _copy_mapping(self.candidate_config)
        self.rationale = _copy_mapping(self.rationale)
        self.metadata = _copy_mapping(self.metadata)


@runtime_checkable
class Planner(Protocol):
    """Boundary contract for turning task intent into a candidate pipeline."""

    planner_id: str

    def plan_result(
        self,
        request: PlannerRequest,
        operator_registry: OperatorRegistry,
    ) -> PlanningResult:
        """Return a candidate plan without executing operators or persisting data."""
