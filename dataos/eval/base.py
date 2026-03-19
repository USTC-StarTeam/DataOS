"""Evaluation interfaces and handoff objects for DataOS."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dataos.core.dataset import DatasetVersion
from dataos.core.eval import EvalResult
from dataos.core.task import TaskSpec

__all__ = [
    "EvaluationRequest",
    "Evaluator",
    "EvaluatorSpec",
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


@dataclass(frozen=True, slots=True)
class EvaluatorSpec:
    """Static metadata describing a reusable evaluator implementation."""

    evaluator_id: str
    name: str
    metrics: list[str] = field(default_factory=list)
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.evaluator_id, "evaluator_id")
        _require_text(self.name, "name")
        object.__setattr__(self, "metrics", list(dict.fromkeys(self.metrics)))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata))


@dataclass(slots=True)
class EvaluationRequest:
    """Stable input handed from controller/runtime outputs into the eval layer."""

    task_spec: TaskSpec
    dataset: DatasetVersion
    baseline: DatasetVersion | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.artifacts = _copy_mapping(self.artifacts)
        self.metadata = _copy_mapping(self.metadata)


@runtime_checkable
class Evaluator(Protocol):
    """Boundary contract for candidate-version evaluation."""

    spec: EvaluatorSpec

    def evaluate(self, request: EvaluationRequest) -> EvalResult:
        """Evaluate one candidate dataset version and return structured feedback."""
