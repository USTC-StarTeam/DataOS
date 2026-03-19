"""Sequential runtime executor for DataOS pipelines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from inspect import signature
from time import perf_counter
from traceback import format_exc
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from dataos.core import Operator, OperatorContext, OperatorResult, PipelineStep, StepStatus


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(dict(value))


def _copy_list(value: Sequence[Any] | None) -> list[Any]:
    if value is None:
        return []
    return deepcopy(list(value))


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


def _clone_step(step: PipelineStep) -> PipelineStep:
    return PipelineStep(
        step_id=step.step_id,
        operator_id=step.operator_id,
        inputs=step.inputs,
        config=step.config,
        outputs=step.outputs,
        status=step.status,
        metrics=step.metrics,
    )


@runtime_checkable
class OperatorRegistryLike(Protocol):
    """Minimal registry surface required by the sequential executor."""

    def get_operator(self, operator_id: str) -> Operator:
        """Return the operator registered for the given id."""


@runtime_checkable
class RunStoreLike(Protocol):
    """Optional persistence hooks consumed by the sequential executor."""

    def save_plan(
        self,
        run_id: str,
        pipeline: Sequence[PipelineStep],
        task_spec: Any | None = None,
    ) -> None:
        """Persist the run-level plan before execution starts."""

    def save_step(
        self,
        run_id: str,
        step: PipelineStep,
        record: "StepExecutionRecord",
    ) -> None:
        """Persist a completed step snapshot and its execution record."""

    def save_result(self, run_id: str, result: "PipelineExecutionResult") -> None:
        """Persist the final run result."""


@dataclass(slots=True)
class StepExecutionRecord:
    """Observed runtime data for one executed pipeline step."""

    step_id: str
    operator_id: str
    input_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    inputs: list[Any] = field(default_factory=list)
    outputs: list[Any] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.step_id, "step_id")
        _require_text(self.operator_id, "operator_id")
        self.input_refs = _copy_list(self.input_refs)
        self.output_refs = _copy_list(self.output_refs)
        self.inputs = _copy_list(self.inputs)
        self.outputs = _copy_list(self.outputs)
        self.config = _copy_mapping(self.config)
        self.metrics = _copy_mapping(self.metrics)
        self.artifacts = _copy_mapping(self.artifacts)
        self.metadata = _copy_mapping(self.metadata)
        self.error = None if self.error is None else _copy_mapping(self.error)
        if isinstance(self.status, str):
            self.status = StepStatus(self.status)
        if self.duration_seconds is not None:
            self.duration_seconds = float(self.duration_seconds)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class PipelineExecutionResult:
    """Aggregated result for one sequential pipeline run."""

    run_id: str
    status: StepStatus
    steps: list[PipelineStep] = field(default_factory=list)
    step_records: list[StepExecutionRecord] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_text(self.run_id, "run_id")
        if isinstance(self.status, str):
            self.status = StepStatus(self.status)
        self.steps = [_clone_step(step) for step in self.steps]
        self.step_records = [
            record if isinstance(record, StepExecutionRecord) else StepExecutionRecord(**record)
            for record in self.step_records
        ]
        self.state = deepcopy(self.state)
        self.metrics = _copy_mapping(self.metrics)
        self.error = None if self.error is None else _copy_mapping(self.error)
        if self.duration_seconds is not None:
            self.duration_seconds = float(self.duration_seconds)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    def get_step_record(self, step_id: str) -> StepExecutionRecord:
        for record in self.step_records:
            if record.step_id == step_id:
                return record
        raise KeyError(f"step record '{step_id}' not found")


@dataclass(slots=True)
class RuntimeExecutionRequest:
    """Structured runtime input that keeps planner/controller handoff explicit."""

    pipeline: list[PipelineStep] = field(default_factory=list)
    initial_inputs: dict[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    task_spec: Any | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pipeline = [_clone_step(step) for step in self.pipeline]
        self.initial_inputs = _copy_mapping(self.initial_inputs)
        self.resources = _copy_mapping(self.resources)
        self.metadata = _copy_mapping(self.metadata)


class PipelineExecutionError(RuntimeError):
    """Raised when a pipeline run fails and caller requested an exception."""

    def __init__(
        self,
        run_id: str,
        step_id: str,
        operator_id: str,
        message: str,
        *,
        result: PipelineExecutionResult,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.step_id = step_id
        self.operator_id = operator_id
        self.result = result


class SequentialRuntimeExecutor:
    """Execute a pipeline step-by-step using registered operators."""

    def __init__(
        self,
        operator_registry: OperatorRegistryLike | Mapping[str, Operator],
        *,
        run_store: RunStoreLike | None = None,
        resources: Mapping[str, Any] | None = None,
    ) -> None:
        self._operator_registry = operator_registry
        self._run_store = run_store
        self._resources = _copy_mapping(resources)

    def execute(
        self,
        pipeline: Sequence[PipelineStep],
        *,
        initial_inputs: Mapping[str, Any] | None = None,
        run_id: str | None = None,
        task_spec: Any | None = None,
        resources: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        raise_on_error: bool = False,
    ) -> PipelineExecutionResult:
        cloned_steps = [_clone_step(step) for step in pipeline]
        resolved_run_id = run_id or f"run-{uuid4().hex[:12]}"
        state = deepcopy(dict(initial_inputs or {}))
        shared_resources = {**self._resources, **dict(resources or {})}
        shared_metadata = dict(metadata or {})
        run_started_at = _utc_now_iso()
        run_started_clock = perf_counter()
        step_records: list[StepExecutionRecord] = []
        run_error: dict[str, Any] | None = None

        self._initialize_run(resolved_run_id, cloned_steps, task_spec, shared_metadata)

        for step in cloned_steps:
            step.status = StepStatus.RUNNING
            started_at = _utc_now_iso()
            started_clock = perf_counter()
            try:
                operator = self._resolve_operator(step.operator_id)
                inputs = self._resolve_inputs(step, state)
                context = OperatorContext(
                    run_id=resolved_run_id,
                    step_id=step.step_id,
                    inputs=inputs,
                    config=step.config,
                    task_spec=task_spec,
                    resources=shared_resources,
                    metadata={
                        **shared_metadata,
                        "input_refs": list(step.inputs),
                        "output_refs": list(step.outputs),
                        "operator_id": step.operator_id,
                    },
                )
                result = operator.execute(context)
                if not isinstance(result, OperatorResult):
                    raise TypeError(
                        f"operator '{step.operator_id}' must return OperatorResult, got {type(result).__name__}"
                    )

                finished_at = _utc_now_iso()
                duration_seconds = perf_counter() - started_clock
                output_refs = self._store_outputs(step, result.outputs, state)
                step.status = StepStatus.SUCCEEDED
                step.metrics = deepcopy(result.metrics)
                record = StepExecutionRecord(
                    step_id=step.step_id,
                    operator_id=step.operator_id,
                    input_refs=step.inputs,
                    output_refs=output_refs,
                    inputs=inputs,
                    outputs=result.outputs,
                    config=step.config,
                    metrics=result.metrics,
                    artifacts=result.artifacts,
                    metadata=result.metadata,
                    status=StepStatus.SUCCEEDED,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=duration_seconds,
                )
                step_records.append(record)
                self._save_step(resolved_run_id, step, record)
            except Exception as exc:
                finished_at = _utc_now_iso()
                duration_seconds = perf_counter() - started_clock
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": format_exc(),
                }
                step.status = StepStatus.FAILED
                step.metrics = {}
                record = StepExecutionRecord(
                    step_id=step.step_id,
                    operator_id=step.operator_id,
                    input_refs=step.inputs,
                    output_refs=step.outputs,
                    inputs=self._safe_resolve_inputs(step, state),
                    outputs=[],
                    config=step.config,
                    metrics={},
                    artifacts={},
                    metadata={},
                    status=StepStatus.FAILED,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_seconds=duration_seconds,
                    error=error,
                )
                step_records.append(record)
                self._save_step(resolved_run_id, step, record)
                run_error = {
                    "step_id": step.step_id,
                    "operator_id": step.operator_id,
                    **error,
                }
                break

        run_finished_at = _utc_now_iso()
        duration_seconds = perf_counter() - run_started_clock
        status = StepStatus.FAILED if run_error else StepStatus.SUCCEEDED
        run_metrics = self._build_run_metrics(cloned_steps, step_records, duration_seconds)
        result = PipelineExecutionResult(
            run_id=resolved_run_id,
            task_id=getattr(task_spec, "task_id", None),
            status=status,
            steps=cloned_steps,
            step_records=step_records,
            state=state,
            metrics=run_metrics,
            started_at=run_started_at,
            finished_at=run_finished_at,
            duration_seconds=duration_seconds,
            error=run_error,
        )
        self._save_result(resolved_run_id, result)

        if run_error and raise_on_error:
            raise PipelineExecutionError(
                resolved_run_id,
                run_error["step_id"],
                run_error["operator_id"],
                run_error["message"],
                result=result,
            )

        return result

    def execute_request(
        self,
        request: RuntimeExecutionRequest,
        *,
        raise_on_error: bool = False,
    ) -> PipelineExecutionResult:
        """Execute a structured request object without changing executor semantics."""

        return self.execute(
            request.pipeline,
            initial_inputs=request.initial_inputs,
            run_id=request.run_id,
            task_spec=request.task_spec,
            resources=request.resources,
            metadata=request.metadata,
            raise_on_error=raise_on_error,
        )

    def _resolve_operator(self, operator_id: str) -> Operator:
        registry = self._operator_registry
        operator: Any | None = None

        if isinstance(registry, Mapping):
            operator = registry.get(operator_id)
        elif hasattr(registry, "get_operator"):
            operator = registry.get_operator(operator_id)
        elif hasattr(registry, "get"):
            operator = registry.get(operator_id)

        if operator is None:
            raise KeyError(f"operator '{operator_id}' is not registered")
        if not isinstance(operator, Operator):
            raise TypeError(f"registered object for '{operator_id}' does not implement Operator")
        return operator

    def _resolve_inputs(self, step: PipelineStep, state: Mapping[str, Any]) -> list[Any]:
        resolved_inputs: list[Any] = []
        for input_ref in step.inputs:
            if input_ref not in state:
                raise KeyError(f"step '{step.step_id}' is missing input ref '{input_ref}'")
            value = deepcopy(state[input_ref])
            if isinstance(value, list):
                resolved_inputs.extend(value)
            else:
                resolved_inputs.append(value)
        return resolved_inputs

    def _safe_resolve_inputs(self, step: PipelineStep, state: Mapping[str, Any]) -> list[Any]:
        try:
            return self._resolve_inputs(step, state)
        except Exception:
            return []

    def _store_outputs(
        self,
        step: PipelineStep,
        outputs: Sequence[Any],
        state: dict[str, Any],
    ) -> list[str]:
        resolved_outputs = deepcopy(list(outputs))
        output_refs = list(step.outputs) if step.outputs else [step.step_id]

        if not step.outputs:
            step.outputs = list(output_refs)

        if len(output_refs) == 1:
            state[output_refs[0]] = resolved_outputs
            return output_refs

        if len(output_refs) != len(resolved_outputs):
            raise ValueError(
                f"step '{step.step_id}' declares {len(output_refs)} outputs but produced {len(resolved_outputs)}"
            )

        for output_ref, value in zip(output_refs, resolved_outputs):
            state[output_ref] = value
        return output_refs

    def _build_run_metrics(
        self,
        steps: Sequence[PipelineStep],
        step_records: Sequence[StepExecutionRecord],
        duration_seconds: float,
    ) -> dict[str, Any]:
        succeeded_steps = sum(record.status == StepStatus.SUCCEEDED for record in step_records)
        failed_steps = sum(record.status == StepStatus.FAILED for record in step_records)
        return {
            "planned_steps": len(steps),
            "completed_steps": len(step_records),
            "succeeded_steps": succeeded_steps,
            "failed_steps": failed_steps,
            "duration_seconds": duration_seconds,
            "step_durations": {
                record.step_id: record.duration_seconds for record in step_records if record.duration_seconds is not None
            },
        }

    def _save_plan(
        self,
        run_id: str,
        pipeline: Sequence[PipelineStep],
        task_spec: Any | None,
    ) -> None:
        self._call_store_method(
            "save_plan",
            ((run_id, pipeline, task_spec), {}),
            ((run_id, pipeline), {}),
            strict=self._has_store_method("save_plan"),
        )

    def _save_step(self, run_id: str, step: PipelineStep, record: StepExecutionRecord) -> None:
        if self._call_store_method(
            "save_step",
            ((run_id, step, record), {}),
            ((run_id, step), {}),
            strict=self._has_store_method("save_step"),
        ):
            return
        self._call_store_method(
            "update_metadata",
            ((run_id, {"runtime": {"step_records": {record.step_id: record.to_dict()}}}), {}),
        )

    def _save_result(self, run_id: str, result: PipelineExecutionResult) -> None:
        if self._call_store_method(
            "save_result",
            ((run_id, result), {}),
            strict=self._has_store_method("save_result"),
        ):
            return
        self._call_store_method(
            "update_metrics",
            ((run_id, {"runtime": result.metrics}), {}),
        )
        self._call_store_method(
            "update_metadata",
            ((run_id, self._build_runtime_summary(result)), {}),
        )

    def _initialize_run(
        self,
        run_id: str,
        pipeline: Sequence[PipelineStep],
        task_spec: Any | None,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        run_metadata = _copy_mapping(metadata)
        task_id = getattr(task_spec, "task_id", None)
        if task_id is not None and "task_id" not in run_metadata:
            run_metadata["task_id"] = task_id

        created = self._call_store_method(
            "create_run",
            ((run_id,), {"plan": pipeline, "metadata": run_metadata or None}),
        )
        if created:
            return

        self._save_plan(run_id, pipeline, task_spec)
        if run_metadata:
            self._call_store_method("update_metadata", ((run_id, run_metadata), {}))

    def _build_runtime_summary(self, result: PipelineExecutionResult) -> dict[str, Any]:
        runtime_summary: dict[str, Any] = {
            "status": result.status.value,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "duration_seconds": result.duration_seconds,
            "step_records": {
                record.step_id: record.to_dict()
                for record in result.step_records
            },
        }
        if result.task_id is not None:
            runtime_summary["task_id"] = result.task_id
        if result.error is not None:
            runtime_summary["error"] = deepcopy(result.error)
        return {"runtime": runtime_summary}

    def _has_store_method(self, method_name: str) -> bool:
        return self._run_store is not None and callable(getattr(self._run_store, method_name, None))

    def _call_store_method(
        self,
        method_name: str,
        *calls: tuple[tuple[Any, ...], dict[str, Any]],
        strict: bool = False,
    ) -> bool:
        if self._run_store is None:
            return False

        method = getattr(self._run_store, method_name, None)
        if not callable(method):
            return False

        for args, kwargs in calls:
            if self._supports_method_call(method, *args, **kwargs):
                method(*args, **kwargs)
                return True

        if strict:
            raise TypeError(
                f"run_store.{method_name} does not support executor-compatible arguments"
            )
        return False

    def _supports_method_call(self, method: Any, /, *args: Any, **kwargs: Any) -> bool:
        try:
            signature(method).bind(*args, **kwargs)
        except (TypeError, ValueError):
            return False
        return True


RuntimeExecutor = SequentialRuntimeExecutor
RuntimeExecutionResult = PipelineExecutionResult

__all__ = [
    "OperatorRegistryLike",
    "PipelineExecutionError",
    "PipelineExecutionResult",
    "RuntimeExecutionRequest",
    "RuntimeExecutionResult",
    "RunStoreLike",
    "RuntimeExecutor",
    "SequentialRuntimeExecutor",
    "StepExecutionRecord",
]
