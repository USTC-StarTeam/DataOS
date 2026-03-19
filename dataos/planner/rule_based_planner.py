"""Minimal rule-based planner for the DataOS SFT MVP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from dataos.core import PipelineStep, TaskSpec, TaskType
from dataos.planner.base import PlannerRequest, PlanningResult
from dataos.registry.operator_registry import OperatorRegistry

__all__ = ["PlanningError", "RuleBasedPlanner"]


class PlanningError(ValueError):
    """Raised when a task cannot be converted into an executable rule-based plan."""


class RuleBasedPlanner:
    """Map a ``TaskSpec`` into a minimal ordered ``PipelineStep`` sequence."""

    planner_id = "rule_based"

    _TASK_TEMPLATES: dict[TaskType, tuple[str, ...]] = {
        TaskType.CLEAN: ("filter_by_rule", "dedup"),
        TaskType.SELECT: ("filter_by_score", "sample_by_strategy"),
        TaskType.EVALUATE: (),
        TaskType.ITERATE: (
            "filter_by_rule",
            "dedup",
            "filter_by_score",
            "sample_by_strategy",
        ),
    }

    def plan(self, task_spec: TaskSpec, input_ref: str | None = None) -> list[PipelineStep]:
        """Build a minimal executable plan for the provided task."""

        if not isinstance(task_spec, TaskSpec):
            raise TypeError("task_spec must be a TaskSpec instance")

        operator_ids = self._resolve_operator_ids(task_spec)
        self._validate_allowed_operators(task_spec, operator_ids)

        upstream_ref = input_ref or f"{task_spec.task_id}:input"
        steps: list[PipelineStep] = []
        total_steps = len(operator_ids)

        for index, operator_id in enumerate(operator_ids, start=1):
            step_id = f"{task_spec.task_type.value}-{index:02d}-{operator_id}"
            output_ref = self._build_output_ref(task_spec, index, total_steps, operator_id)
            steps.append(
                PipelineStep(
                    step_id=step_id,
                    operator_id=operator_id,
                    inputs=[upstream_ref],
                    config=self._build_step_config(task_spec, operator_id),
                    outputs=[output_ref],
                )
            )
            upstream_ref = output_ref

        return steps

    def plan_result(
        self,
        request: PlannerRequest,
        operator_registry: OperatorRegistry,
    ) -> PlanningResult:
        """Return the rule-based plan wrapped in the planner boundary object."""

        input_ref = request.datasets[0].ref if request.datasets else None
        pipeline = self.plan(request.task_spec, input_ref=input_ref)
        self._validate_registered_operators(operator_registry, pipeline)
        return PlanningResult(
            plan_id=f"{request.task_spec.task_id}:{self.planner_id}",
            planner_id=self.planner_id,
            pipeline=pipeline,
            candidate_config={"strategy": self.planner_id},
            rationale={
                "task_type": request.task_spec.task_type.value,
                "allowed_operators": list(request.task_spec.allowed_operators),
                "runtime_only_pipeline": True,
                "evaluation": {
                    "handoff": "post_runtime",
                    "evaluator_id": request.task_spec.eval_spec.evaluator_id,
                    **self._build_eval_config(request.task_spec),
                },
            },
            metadata=request.metadata,
        )

    def _validate_registered_operators(
        self,
        operator_registry: OperatorRegistry,
        pipeline: Sequence[PipelineStep],
    ) -> None:
        # Allow early planning before the runtime registry is wired, but validate
        # operator availability once a registry has been populated.
        if len(operator_registry) == 0:
            return

        missing: list[str] = []
        for step in pipeline:
            if operator_registry.supports(step.operator_id):
                continue
            if step.operator_id not in missing:
                missing.append(step.operator_id)

        if missing:
            missing_list = ", ".join(missing)
            raise PlanningError(
                "rule-based planner cannot emit an executable plan because required "
                f"operators are not registered: {missing_list}"
            )

    def _resolve_operator_ids(self, task_spec: TaskSpec) -> list[str]:
        try:
            template = self._TASK_TEMPLATES[task_spec.task_type]
        except KeyError as exc:
            supported = ", ".join(task_type.value for task_type in self._TASK_TEMPLATES)
            raise PlanningError(
                f"rule-based planner does not support task_type={task_spec.task_type.value!r}; "
                f"supported task types: {supported}"
            ) from exc

        return list(template)

    def _validate_allowed_operators(self, task_spec: TaskSpec, operator_ids: Sequence[str]) -> None:
        if not task_spec.allowed_operators:
            return

        allowed = set(task_spec.allowed_operators)
        missing = [operator_id for operator_id in operator_ids if operator_id not in allowed]
        if missing:
            missing_list = ", ".join(missing)
            raise PlanningError(
                f"task {task_spec.task_id!r} cannot be planned because required operators are "
                f"not allowed: {missing_list}"
            )

    def _build_output_ref(
        self,
        task_spec: TaskSpec,
        step_index: int,
        total_steps: int,
        operator_id: str,
    ) -> str:
        if step_index == total_steps:
            return f"{task_spec.task_id}:output"
        return f"{task_spec.task_id}:step:{step_index:02d}:{operator_id}:output"

    def _build_step_config(self, task_spec: TaskSpec, operator_id: str) -> dict[str, Any]:
        config: dict[str, Any] = {
            "task_id": task_spec.task_id,
            "task_type": task_spec.task_type.value,
            "training_stage": task_spec.training_stage.value,
        }

        if operator_id == "filter_by_rule":
            config.update(self._build_rule_filter_config(task_spec))
        elif operator_id == "dedup":
            config.update(self._build_dedup_config(task_spec))
        elif operator_id == "filter_by_score":
            config.update(self._build_score_filter_config(task_spec))
        elif operator_id == "sample_by_strategy":
            config.update(self._build_sampling_config(task_spec))
        else:
            config["operator_config"] = deepcopy(task_spec.metadata.get(operator_id, {}))

        return config

    def _build_rule_filter_config(self, task_spec: TaskSpec) -> dict[str, Any]:
        return {
            "required_fields": self._as_text_list(task_spec.constraints.get("required_fields")),
            "forbidden_content": self._as_text_list(task_spec.constraints.get("forbidden_content")),
            "quality_gates": self._as_mapping(task_spec.constraints.get("quality_gates")),
        }

    def _build_dedup_config(self, task_spec: TaskSpec) -> dict[str, Any]:
        schema_fields = self._schema_field_names(task_spec.input_schema)
        dedup_keys = self._as_text_list(task_spec.constraints.get("dedup_keys"))

        if not dedup_keys:
            if "id" in schema_fields:
                dedup_keys.append("id")
            if {"prompt", "response"}.issubset(schema_fields):
                dedup_keys.append("prompt_response_normalized")
            elif "messages" in schema_fields:
                dedup_keys.append("messages_normalized")

        if not dedup_keys:
            dedup_keys = ["record_fingerprint"]

        return {"dedup_keys": dedup_keys}

    def _build_score_filter_config(self, task_spec: TaskSpec) -> dict[str, Any]:
        score_field = self._infer_score_field(task_spec)
        threshold = self._infer_score_threshold(task_spec)
        direction = str(task_spec.objective.get("direction", "maximize"))
        return {
            "score_field": score_field,
            "threshold": threshold,
            "direction": direction,
        }

    def _build_sampling_config(self, task_spec: TaskSpec) -> dict[str, Any]:
        strategy = str(task_spec.constraints.get("sampling_strategy", "random"))
        group_by = self._as_text_list(task_spec.constraints.get("sample_group_by"))
        random_seed = task_spec.metadata.get("random_seed", 0)
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            random_seed = 0

        return {
            "strategy": strategy,
            "max_samples": task_spec.budget.max_samples,
            "group_by": group_by,
            "random_seed": random_seed,
        }

    def _build_eval_config(self, task_spec: TaskSpec) -> dict[str, Any]:
        return {
            "metrics": list(task_spec.eval_spec.metrics),
            "dataset_split": task_spec.eval_spec.dataset_split,
            "artifacts": list(task_spec.eval_spec.artifacts),
            "success_criteria": deepcopy(task_spec.eval_spec.success_criteria),
            "objective": deepcopy(task_spec.objective),
        }

    def _infer_score_field(self, task_spec: TaskSpec) -> str:
        explicit = task_spec.constraints.get("score_field")
        if isinstance(explicit, str) and explicit.strip():
            return explicit

        schema_fields = self._schema_field_names(task_spec.input_schema)
        for candidate in ("score", "quality_score"):
            if candidate in schema_fields:
                return candidate
        return "score"

    def _infer_score_threshold(self, task_spec: TaskSpec) -> float:
        candidates = (
            task_spec.constraints.get("score_threshold"),
            task_spec.constraints.get("min_score"),
            task_spec.objective.get("score_threshold"),
            task_spec.objective.get("min_score"),
        )

        for candidate in candidates:
            if isinstance(candidate, bool):
                continue
            if isinstance(candidate, (int, float)):
                return float(candidate)
        return 0.0

    def _schema_field_names(self, schema: Mapping[str, Any]) -> set[str]:
        fields = schema.get("fields", [])
        if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
            return set()

        names: set[str] = set()
        for field_spec in fields:
            if isinstance(field_spec, Mapping):
                name = field_spec.get("name")
                if isinstance(name, str) and name.strip():
                    names.add(name)
        return names

    def _as_mapping(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        return deepcopy(dict(value))

    def _as_text_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
            return []

        items: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text and text not in items:
                    items.append(text)
        return items
