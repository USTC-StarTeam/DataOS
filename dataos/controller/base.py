"""Controller interfaces and orchestration handoff objects for DataOS."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dataos.core.eval import EvalResult
from dataos.core.optimization import OptimizationState
from dataos.core.task import TaskSpec
from dataos.eval.result_analyzer import AnalysisSummary
from dataos.planner.base import PlanningResult
from dataos.runtime.executor import RuntimeExecutionResult

__all__ = [
    "Controller",
    "ControllerDecision",
    "ControllerRoundContext",
]


def _copy_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(value)


@dataclass(slots=True)
class ControllerRoundContext:
    """Controller-visible snapshot of one optimization round."""

    task_spec: TaskSpec
    optimization_state: OptimizationState
    latest_plan: PlanningResult | None = None
    latest_runtime: RuntimeExecutionResult | None = None
    latest_evals: list[EvalResult] = field(default_factory=list)
    analysis: AnalysisSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.latest_evals = list(self.latest_evals)
        self.metadata = _copy_mapping(self.metadata)


@dataclass(slots=True)
class ControllerDecision:
    """Controller output describing whether and how the loop should continue."""

    should_continue: bool
    next_action: str = "stop"
    selected_plan: PlanningResult | None = None
    updated_state: OptimizationState | None = None
    best_result: EvalResult | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.next_action:
            raise ValueError("next_action must not be empty")
        self.metadata = _copy_mapping(self.metadata)


@runtime_checkable
class Controller(Protocol):
    """Boundary contract for the outer optimization loop coordinator."""

    controller_id: str

    def decide(self, context: ControllerRoundContext) -> ControllerDecision:
        """Decide whether to continue, stop, or re-plan the next optimization round."""
