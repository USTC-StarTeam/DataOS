"""Declarative task schema objects shared by planners and experiment configs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any

from dataos.core.enums import IntelligenceLevel, TaskType, TrainingStage


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _copy_mapping(value: Mapping[str, Any] | None, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return deepcopy(dict(value))


def _copy_text_list(value: Sequence[str] | None, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")

    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} must contain only strings")
        _require_text(item, field_name)
        items.append(item)
    return list(dict.fromkeys(items))


def _coerce_enum(
    value: TrainingStage | IntelligenceLevel | TaskType | str,
    enum_type: type[TrainingStage] | type[IntelligenceLevel] | type[TaskType],
    field_name: str,
) -> TrainingStage | IntelligenceLevel | TaskType:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as exc:
        valid_values = ", ".join(member.value for member in enum_type)
        raise ValueError(f"{field_name} must be one of: {valid_values}") from exc


def _coerce_non_negative_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _coerce_non_negative_number(value: float | int | None, field_name: str) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field_info.name: _serialize(getattr(value, field_info.name)) for field_info in fields(value)}
    if isinstance(value, Mapping):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return deepcopy(value)


@dataclass(slots=True)
class BudgetSpec:
    """Execution budget limits that bound the search or runtime surface."""

    max_iterations: int | None = None
    max_samples: int | None = None
    max_model_calls: int | None = None
    max_tokens: int | None = None
    max_runtime_seconds: int | None = None
    max_cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.max_iterations = _coerce_non_negative_int(self.max_iterations, "max_iterations")
        self.max_samples = _coerce_non_negative_int(self.max_samples, "max_samples")
        self.max_model_calls = _coerce_non_negative_int(self.max_model_calls, "max_model_calls")
        self.max_tokens = _coerce_non_negative_int(self.max_tokens, "max_tokens")
        self.max_runtime_seconds = _coerce_non_negative_int(
            self.max_runtime_seconds, "max_runtime_seconds"
        )
        max_cost = _coerce_non_negative_number(self.max_cost_usd, "max_cost_usd")
        self.max_cost_usd = None if max_cost is None else float(max_cost)
        self.metadata = _copy_mapping(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BudgetSpec":
        return cls(
            max_iterations=data.get("max_iterations"),
            max_samples=data.get("max_samples"),
            max_model_calls=data.get("max_model_calls"),
            max_tokens=data.get("max_tokens"),
            max_runtime_seconds=data.get("max_runtime_seconds"),
            max_cost_usd=data.get("max_cost_usd"),
            metadata=_copy_mapping(data.get("metadata"), "metadata"),
        )


@dataclass(slots=True)
class StopCondition:
    """Declarative conditions that tell the controller when to stop iterating."""

    metric_thresholds: dict[str, float] = field(default_factory=dict)
    max_rounds_without_improvement: int | None = None
    min_improvement_delta: float | None = None
    stop_on_budget_exhaustion: bool = True
    custom_rules: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metric_thresholds = _copy_mapping(self.metric_thresholds, "metric_thresholds")
        self.max_rounds_without_improvement = _coerce_non_negative_int(
            self.max_rounds_without_improvement,
            "max_rounds_without_improvement",
        )
        min_delta = _coerce_non_negative_number(self.min_improvement_delta, "min_improvement_delta")
        self.min_improvement_delta = None if min_delta is None else float(min_delta)
        self.custom_rules = _copy_text_list(self.custom_rules, "custom_rules")
        self.metadata = _copy_mapping(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StopCondition":
        return cls(
            metric_thresholds=_copy_mapping(data.get("metric_thresholds"), "metric_thresholds"),
            max_rounds_without_improvement=data.get("max_rounds_without_improvement"),
            min_improvement_delta=data.get("min_improvement_delta"),
            stop_on_budget_exhaustion=bool(data.get("stop_on_budget_exhaustion", True)),
            custom_rules=_copy_text_list(data.get("custom_rules"), "custom_rules"),
            metadata=_copy_mapping(data.get("metadata"), "metadata"),
        )


@dataclass(slots=True)
class EvalSpec:
    """Evaluation requirements that stay stable across planner implementations."""

    evaluator_id: str
    metrics: list[str] = field(default_factory=list)
    dataset_split: str | None = None
    artifacts: list[str] = field(default_factory=list)
    success_criteria: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.evaluator_id, "evaluator_id")
        if self.dataset_split is not None:
            _require_text(self.dataset_split, "dataset_split")
        self.metrics = _copy_text_list(self.metrics, "metrics")
        self.artifacts = _copy_text_list(self.artifacts, "artifacts")
        self.success_criteria = _copy_mapping(self.success_criteria, "success_criteria")
        self.metadata = _copy_mapping(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalSpec":
        return cls(
            evaluator_id=str(data["evaluator_id"]),
            metrics=_copy_text_list(data.get("metrics"), "metrics"),
            dataset_split=data.get("dataset_split"),
            artifacts=_copy_text_list(data.get("artifacts"), "artifacts"),
            success_criteria=_copy_mapping(data.get("success_criteria"), "success_criteria"),
            metadata=_copy_mapping(data.get("metadata"), "metadata"),
        )


@dataclass(slots=True)
class TaskSpec:
    """Declarative task description consumed by planners, runtimes, and experiments."""

    task_id: str
    name: str
    training_stage: TrainingStage | str
    intelligence_level: IntelligenceLevel | str
    task_type: TaskType | str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    eval_spec: EvalSpec | dict[str, Any]
    constraints: dict[str, Any] = field(default_factory=dict)
    objective: dict[str, Any] = field(default_factory=dict)
    budget: BudgetSpec | dict[str, Any] = field(default_factory=BudgetSpec)
    stop_condition: StopCondition | dict[str, Any] = field(default_factory=StopCondition)
    allowed_operators: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.task_id, "task_id")
        _require_text(self.name, "name")
        self.training_stage = _coerce_enum(
            self.training_stage,
            TrainingStage,
            "training_stage",
        )
        self.intelligence_level = _coerce_enum(
            self.intelligence_level,
            IntelligenceLevel,
            "intelligence_level",
        )
        self.task_type = _coerce_enum(self.task_type, TaskType, "task_type")
        self.input_schema = _copy_mapping(self.input_schema, "input_schema")
        self.output_schema = _copy_mapping(self.output_schema, "output_schema")
        if not self.input_schema:
            raise ValueError("input_schema must not be empty")
        if not self.output_schema:
            raise ValueError("output_schema must not be empty")
        self.constraints = _copy_mapping(self.constraints, "constraints")
        self.objective = _copy_mapping(self.objective, "objective")
        if isinstance(self.budget, BudgetSpec):
            self.budget = BudgetSpec.from_dict(self.budget.to_dict())
        else:
            self.budget = BudgetSpec.from_dict(self.budget)
        if isinstance(self.stop_condition, StopCondition):
            self.stop_condition = StopCondition.from_dict(self.stop_condition.to_dict())
        else:
            self.stop_condition = StopCondition.from_dict(self.stop_condition)
        if isinstance(self.eval_spec, EvalSpec):
            self.eval_spec = EvalSpec.from_dict(self.eval_spec.to_dict())
        else:
            self.eval_spec = EvalSpec.from_dict(self.eval_spec)
        self.allowed_operators = _copy_text_list(self.allowed_operators, "allowed_operators")
        self.metadata = _copy_mapping(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of the task declaration."""

        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        return cls(
            task_id=str(data["task_id"]),
            name=str(data["name"]),
            training_stage=data["training_stage"],
            intelligence_level=data["intelligence_level"],
            task_type=data["task_type"],
            input_schema=_copy_mapping(data.get("input_schema"), "input_schema"),
            output_schema=_copy_mapping(data.get("output_schema"), "output_schema"),
            constraints=_copy_mapping(data.get("constraints"), "constraints"),
            objective=_copy_mapping(data.get("objective"), "objective"),
            budget=data.get("budget", {}),
            stop_condition=data.get("stop_condition", {}),
            eval_spec=data["eval_spec"],
            allowed_operators=_copy_text_list(
                data.get("allowed_operators"),
                "allowed_operators",
            ),
            metadata=_copy_mapping(data.get("metadata"), "metadata"),
        )


__all__ = ["BudgetSpec", "EvalSpec", "StopCondition", "TaskSpec"]
