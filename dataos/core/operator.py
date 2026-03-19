"""Operator abstractions shared by DataOS registries and runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

try:
    from dataos.core.enums import OperatorCategory
except ImportError:  # pragma: no cover - defined in a later foundation task.
    OperatorCategory = str  # type: ignore[misc,assignment]


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """Static metadata describing how an operator can be planned and executed."""

    operator_id: str
    name: str
    category: OperatorCategory | str
    input_type: str
    output_type: str
    config_schema: dict[str, Any] = field(default_factory=dict)
    deterministic: bool = True

    def __post_init__(self) -> None:
        _require_text(self.operator_id, "operator_id")
        _require_text(self.name, "name")
        _require_text(self.input_type, "input_type")
        _require_text(self.output_type, "output_type")
        if isinstance(self.category, str):
            _require_text(self.category, "category")
        object.__setattr__(self, "config_schema", dict(self.config_schema))


@dataclass(slots=True)
class OperatorContext:
    """Runtime inputs required for executing a single operator step."""

    run_id: str
    step_id: str
    inputs: list[Any] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    task_spec: Any | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        _require_text(self.step_id, "step_id")
        self.inputs = list(self.inputs)
        self.config = dict(self.config)
        self.resources = dict(self.resources)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class OperatorResult:
    """Structured output emitted by an operator execution."""

    outputs: list[Any] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.outputs = list(self.outputs)
        self.metrics = dict(self.metrics)
        self.artifacts = dict(self.artifacts)
        self.metadata = dict(self.metadata)


@runtime_checkable
class Operator(Protocol):
    """Unified operator contract consumed by registry and runtime layers."""

    spec: OperatorSpec

    def execute(self, context: OperatorContext) -> OperatorResult:
        """Execute one operator step and return structured outputs."""

