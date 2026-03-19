"""Evaluation objects shared by evaluators and storage."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any


def _copy_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(value)


@dataclass(slots=True)
class EvalResult:
    """Structured evaluation output for one dataset/task pair."""

    eval_id: str
    dataset_version: str
    task_id: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.eval_id:
            raise ValueError("eval_id must not be empty")
        if not self.dataset_version:
            raise ValueError("dataset_version must not be empty")
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        self.metrics = _copy_mapping(self.metrics)
        self.artifacts = _copy_mapping(self.artifacts)
        self.summary = _copy_mapping(self.summary)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the evaluation result."""

        return asdict(self)


__all__ = ["EvalResult"]
