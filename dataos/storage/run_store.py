"""Run state persistence shared by DataOS controllers and runtimes."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from dataos.core import EvalResult, OptimizationState, PipelineStep


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    return deepcopy(dict(value))


def _deep_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = _copy_mapping(base)
    for key, value in updates.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _coerce_step(step: PipelineStep | Mapping[str, Any]) -> PipelineStep:
    if isinstance(step, PipelineStep):
        return PipelineStep(**step.to_dict())
    return PipelineStep(**dict(step))


def _coerce_steps(steps: Sequence[PipelineStep | Mapping[str, Any]] | None) -> list[PipelineStep]:
    if steps is None:
        return []
    return [_coerce_step(step) for step in steps]


def _coerce_eval_result(result: EvalResult | Mapping[str, Any]) -> EvalResult:
    if isinstance(result, EvalResult):
        return EvalResult(**result.to_dict())
    return EvalResult(**dict(result))


def _coerce_eval_results(
    results: Sequence[EvalResult | Mapping[str, Any]] | None,
) -> list[EvalResult]:
    if results is None:
        return []
    return [_coerce_eval_result(result) for result in results]


def _coerce_optimization_state(
    state: OptimizationState | Mapping[str, Any] | None,
) -> OptimizationState | None:
    if state is None:
        return None
    if isinstance(state, OptimizationState):
        return OptimizationState(**state.to_dict())
    return OptimizationState(**dict(state))


@dataclass(slots=True)
class RunSnapshot:
    """Aggregated run-level view that controller and runtime can both consume."""

    run_id: str
    plan: list[PipelineStep] = field(default_factory=list)
    steps: list[PipelineStep] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    eval_results: list[EvalResult] = field(default_factory=list)
    optimization_state: OptimizationState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        self.plan = _coerce_steps(self.plan)
        self.steps = _coerce_steps(self.steps)
        self.metrics = _copy_mapping(self.metrics)
        self.eval_results = _coerce_eval_results(self.eval_results)
        self.optimization_state = _coerce_optimization_state(self.optimization_state)
        self.metadata = _copy_mapping(self.metadata)

    def get_step(self, step_id: str) -> PipelineStep | None:
        for step in self.steps:
            if step.step_id == step_id:
                return PipelineStep(**step.to_dict())
        return None

    @property
    def latest_eval_result(self) -> EvalResult | None:
        if not self.eval_results:
            return None
        return EvalResult(**self.eval_results[-1].to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "plan": [step.to_dict() for step in self.plan],
            "steps": [step.to_dict() for step in self.steps],
            "metrics": deepcopy(self.metrics),
            "eval_results": [result.to_dict() for result in self.eval_results],
            "optimization_state": (
                None if self.optimization_state is None else self.optimization_state.to_dict()
            ),
            "metadata": deepcopy(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class RunStore:
    """Persist run plans, step state, metrics, and eval outputs as JSON artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def run_path(self, run_id: str) -> Path:
        return self.root / self._path_token(run_id)

    def run_exists(self, run_id: str) -> bool:
        return self.run_path(run_id).exists()

    def list_runs(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            unquote(path.name)
            for path in self.root.iterdir()
            if path.is_dir()
        )

    def create_run(
        self,
        run_id: str,
        *,
        plan: Sequence[PipelineStep | Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        optimization_state: OptimizationState | Mapping[str, Any] | None = None,
    ) -> RunSnapshot:
        state = self._ensure_state_payload(run_id)
        if metadata:
            state["metadata"] = _deep_merge(state.get("metadata", {}), metadata)
        if optimization_state is not None:
            state["optimization_state"] = _coerce_optimization_state(optimization_state).to_dict()
        self._write_state_payload(run_id, state)
        if plan is not None:
            self.save_plan(run_id, plan)
        snapshot = self.load_run(run_id)
        assert snapshot is not None
        return snapshot

    def save(self, snapshot: RunSnapshot) -> None:
        """Persist a full run snapshot for checkpoint/resume workflows."""

        current_state = self._ensure_state_payload(snapshot.run_id)
        state = {
            "run_id": snapshot.run_id,
            "metadata": snapshot.metadata,
            "optimization_state": (
                None if snapshot.optimization_state is None else snapshot.optimization_state.to_dict()
            ),
            "created_at": snapshot.created_at or str(current_state.get("created_at", _utc_now_iso())),
            "updated_at": snapshot.updated_at or _utc_now_iso(),
        }
        self._write_state_payload(snapshot.run_id, state)
        self.save_plan(snapshot.run_id, snapshot.plan)
        self._replace_steps(snapshot.run_id, snapshot.steps)
        self.update_metrics(snapshot.run_id, snapshot.metrics, replace=True)
        self._write_eval_results(snapshot.run_id, snapshot.eval_results)

    def load(self, run_id: str) -> RunSnapshot | None:
        """Alias for runtime state store compatibility."""

        return self.load_run(run_id)

    def update_metadata(self, run_id: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
        state = self._ensure_state_payload(run_id)
        state["metadata"] = _deep_merge(state.get("metadata", {}), metadata)
        state = self._write_state_payload(run_id, state)
        return _copy_mapping(state["metadata"])

    def save_plan(
        self,
        run_id: str,
        plan: Sequence[PipelineStep | Mapping[str, Any]],
        task_spec: Any | None = None,
    ) -> list[PipelineStep]:
        steps = _coerce_steps(plan)
        payload = {"run_id": run_id, "steps": [step.to_dict() for step in steps]}
        self._write_json(self._plan_path(run_id), payload)
        state = self._ensure_state_payload(run_id)
        task_id = getattr(task_spec, "task_id", None)
        if task_id:
            state["metadata"] = _deep_merge(state.get("metadata", {}), {"task_id": task_id})
        self._write_state_payload(run_id, state)
        return _coerce_steps(payload["steps"])

    def load_plan(self, run_id: str) -> list[PipelineStep]:
        payload = self._read_json(self._plan_path(run_id))
        if payload is None:
            return []
        return _coerce_steps(payload.get("steps", []))

    def save_step(
        self,
        run_id: str,
        step: PipelineStep | Mapping[str, Any],
        record: Any | None = None,
    ) -> PipelineStep:
        snapshot = _coerce_step(step)
        self._write_json(self._step_path(run_id, snapshot.step_id), snapshot.to_dict())
        if record is not None:
            record_payload = record.to_dict() if hasattr(record, "to_dict") else deepcopy(record)
            self._write_json(self._record_path(run_id, snapshot.step_id), record_payload)
        self._write_state_payload(run_id, self._ensure_state_payload(run_id))
        return PipelineStep(**snapshot.to_dict())

    def load_step(self, run_id: str, step_id: str) -> PipelineStep | None:
        payload = self._read_json(self._step_path(run_id, step_id))
        if payload is None:
            return None
        return PipelineStep(**payload)

    def load_steps(self, run_id: str) -> list[PipelineStep]:
        plan = self.load_plan(run_id)
        steps_dir = self._steps_dir(run_id)
        recorded_steps: list[PipelineStep] = []
        if steps_dir.exists():
            for path in sorted(steps_dir.glob("*.json")):
                payload = self._read_json(path)
                if payload is None:
                    continue
                recorded_steps.append(PipelineStep(**payload))
        if not plan:
            return sorted(recorded_steps, key=lambda step: step.step_id)

        latest_by_id = {step.step_id: step for step in recorded_steps}
        merged_steps: list[PipelineStep] = []
        seen_step_ids: set[str] = set()
        for planned_step in plan:
            merged_steps.append(latest_by_id.get(planned_step.step_id, planned_step))
            seen_step_ids.add(planned_step.step_id)

        extra_steps = sorted(
            (step for step in recorded_steps if step.step_id not in seen_step_ids),
            key=lambda step: step.step_id,
        )
        merged_steps.extend(extra_steps)
        return _coerce_steps(merged_steps)

    def update_metrics(
        self,
        run_id: str,
        metrics: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> dict[str, Any]:
        merged_metrics = _copy_mapping(metrics) if replace else _deep_merge(self.load_metrics(run_id), metrics)
        payload = {"run_id": run_id, "metrics": merged_metrics}
        self._write_json(self._metrics_path(run_id), payload)
        self._write_state_payload(run_id, self._ensure_state_payload(run_id))
        return _copy_mapping(merged_metrics)

    def load_metrics(self, run_id: str) -> dict[str, Any]:
        payload = self._read_json(self._metrics_path(run_id))
        if payload is None:
            return {}
        return _copy_mapping(payload.get("metrics"))

    def save_eval_result(
        self,
        run_id: str,
        result: EvalResult | Mapping[str, Any],
    ) -> list[EvalResult]:
        eval_result = _coerce_eval_result(result)
        existing_results = [
            current
            for current in self.load_eval_results(run_id)
            if current.eval_id != eval_result.eval_id
        ]
        existing_results.append(eval_result)
        payload = {
            "run_id": run_id,
            "latest_eval_id": eval_result.eval_id,
            "results": [item.to_dict() for item in existing_results],
        }
        self._write_json(self._eval_path(run_id), payload)
        self._write_state_payload(run_id, self._ensure_state_payload(run_id))
        return _coerce_eval_results(payload["results"])

    def save_result(self, run_id: str, result: Any) -> None:
        """Persist the final runtime result into the run snapshot surface."""

        steps = getattr(result, "steps", None)
        if steps is not None:
            record_by_step_id = {
                getattr(record, "step_id", ""): record
                for record in getattr(result, "step_records", [])
                if getattr(record, "step_id", "")
            }
            for step in steps:
                self.save_step(run_id, step, record_by_step_id.get(step.step_id))

        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, Mapping):
            self.update_metrics(run_id, metrics, replace=True)

        runtime_metadata: dict[str, Any] = {}
        status = getattr(result, "status", None)
        if status is not None:
            runtime_metadata["status"] = getattr(status, "value", status)
        if getattr(result, "started_at", None):
            runtime_metadata["started_at"] = result.started_at
        if getattr(result, "finished_at", None):
            runtime_metadata["finished_at"] = result.finished_at
        if getattr(result, "duration_seconds", None) is not None:
            runtime_metadata["duration_seconds"] = float(result.duration_seconds)
        if getattr(result, "error", None) is not None:
            runtime_metadata["error"] = deepcopy(result.error)
        if isinstance(getattr(result, "state", None), Mapping):
            runtime_metadata["state_keys"] = sorted(result.state.keys())

        metadata_updates: dict[str, Any] = {}
        task_id = getattr(result, "task_id", None)
        if task_id:
            metadata_updates["task_id"] = task_id
        if runtime_metadata:
            metadata_updates["runtime"] = runtime_metadata
        if metadata_updates:
            self.update_metadata(run_id, metadata_updates)

    def load_eval_results(self, run_id: str) -> list[EvalResult]:
        payload = self._read_json(self._eval_path(run_id))
        if payload is None:
            return []
        return _coerce_eval_results(payload.get("results", []))

    def load_latest_eval_result(self, run_id: str) -> EvalResult | None:
        results = self.load_eval_results(run_id)
        if not results:
            return None
        return EvalResult(**results[-1].to_dict())

    def save_optimization_state(
        self,
        run_id: str,
        optimization_state: OptimizationState | Mapping[str, Any],
    ) -> OptimizationState:
        state = self._ensure_state_payload(run_id)
        snapshot = _coerce_optimization_state(optimization_state)
        assert snapshot is not None
        state["optimization_state"] = snapshot.to_dict()
        self._write_state_payload(run_id, state)
        return OptimizationState(**snapshot.to_dict())

    def load_optimization_state(self, run_id: str) -> OptimizationState | None:
        state = self._read_state_payload(run_id)
        if state is None or state.get("optimization_state") is None:
            return None
        return OptimizationState(**state["optimization_state"])

    def load_run(self, run_id: str) -> RunSnapshot | None:
        if not self.run_exists(run_id):
            return None
        state = self._read_state_payload(run_id) or self._default_state_payload(run_id)
        return RunSnapshot(
            run_id=run_id,
            plan=self.load_plan(run_id),
            steps=self.load_steps(run_id),
            metrics=self.load_metrics(run_id),
            eval_results=self.load_eval_results(run_id),
            optimization_state=state.get("optimization_state"),
            metadata=state.get("metadata"),
            created_at=str(state.get("created_at", _utc_now_iso())),
            updated_at=str(state.get("updated_at", _utc_now_iso())),
        )

    def _default_state_payload(self, run_id: str) -> dict[str, Any]:
        now = _utc_now_iso()
        return {
            "run_id": run_id,
            "metadata": {},
            "optimization_state": None,
            "created_at": now,
            "updated_at": now,
        }

    def _ensure_state_payload(self, run_id: str) -> dict[str, Any]:
        state = self._read_state_payload(run_id)
        if state is not None:
            return state
        return self._default_state_payload(run_id)

    def _read_state_payload(self, run_id: str) -> dict[str, Any] | None:
        return self._read_json(self._state_path(run_id))

    def _write_state_payload(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        state = {
            "run_id": run_id,
            "metadata": _copy_mapping(payload.get("metadata")),
            "optimization_state": (
                None
                if payload.get("optimization_state") is None
                else _coerce_optimization_state(payload.get("optimization_state")).to_dict()
            ),
            "created_at": str(payload.get("created_at") or _utc_now_iso()),
            "updated_at": _utc_now_iso(),
        }
        self._write_json(self._state_path(run_id), state)
        return state

    def _state_path(self, run_id: str) -> Path:
        return self.run_path(run_id) / "state.json"

    def _plan_path(self, run_id: str) -> Path:
        return self.run_path(run_id) / "plan.json"

    def _metrics_path(self, run_id: str) -> Path:
        return self.run_path(run_id) / "metrics.json"

    def _eval_path(self, run_id: str) -> Path:
        return self.run_path(run_id) / "eval.json"

    def _steps_dir(self, run_id: str) -> Path:
        return self.run_path(run_id) / "steps"

    def _step_path(self, run_id: str, step_id: str) -> Path:
        return self._steps_dir(run_id) / f"{self._path_token(step_id)}.json"

    def _records_dir(self, run_id: str) -> Path:
        return self.run_path(run_id) / "records"

    def _record_path(self, run_id: str, step_id: str) -> Path:
        return self._records_dir(run_id) / f"{self._path_token(step_id)}.json"

    def _path_token(self, value: str) -> str:
        if not value:
            raise ValueError("path identity must not be empty")
        return quote(value, safe="")

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _replace_steps(
        self,
        run_id: str,
        steps: Sequence[PipelineStep | Mapping[str, Any]],
    ) -> None:
        steps_dir = self._steps_dir(run_id)
        records_dir = self._records_dir(run_id)
        expected_names = {f"{self._path_token(step.step_id)}.json" for step in _coerce_steps(steps)}

        for directory in (steps_dir, records_dir):
            if not directory.exists():
                continue
            for path in directory.glob("*.json"):
                if path.name not in expected_names:
                    path.unlink()

        for step in steps:
            self.save_step(run_id, step)

    def _write_eval_results(
        self,
        run_id: str,
        results: Sequence[EvalResult | Mapping[str, Any]],
    ) -> None:
        eval_results = _coerce_eval_results(results)
        payload = {
            "run_id": run_id,
            "latest_eval_id": eval_results[-1].eval_id if eval_results else None,
            "results": [result.to_dict() for result in eval_results],
        }
        self._write_json(self._eval_path(run_id), payload)
        self._write_state_payload(run_id, self._ensure_state_payload(run_id))


def default_run_store_path(cwd: str) -> Path:
    return Path(cwd).resolve() / "artifacts" / "runs"


__all__ = ["RunSnapshot", "RunStore", "default_run_store_path"]
