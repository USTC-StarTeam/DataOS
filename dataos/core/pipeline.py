"""Pipeline execution objects shared by planner and runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

from dataos.core.enums import StepStatus

PipelineStepStatus = StepStatus


def _copy_mapping(value: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(value)


@dataclass(slots=True)
class PipelineStep:
    """Structured runtime state for a single pipeline step."""

    step_id: str
    operator_id: str
    inputs: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    status: PipelineStepStatus = StepStatus.PENDING
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.step_id:
            raise ValueError("step_id must not be empty")
        if not self.operator_id:
            raise ValueError("operator_id must not be empty")
        self.inputs = list(self.inputs)
        self.config = _copy_mapping(self.config)
        self.outputs = list(self.outputs)
        self.metrics = _copy_mapping(self.metrics)
        if isinstance(self.status, str):
            self.status = StepStatus(self.status)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the step state."""

        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


__all__ = ["PipelineStep", "PipelineStepStatus"]
