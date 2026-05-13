from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataos.core.artifact import Artifact
from dataos.core.experiment import Experiment
from dataos.core.registry import OperatorRegistry
from dataos.core.task import TaskTemplate
from dataos.llm.client import LLMClient
from dataos.prompts.agent_rules import (
    PLAN_MAX_STEPS,
    build_plan_system_prompt,
    build_plan_user_prompt,
    build_review_system_prompt,
    build_review_user_prompt,
    normalize_review_payload,
    validate_plan_payload,
)


@dataclass(slots=True)
class PlanStep:
    operator_name: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {"operator_name": self.operator_name, "params": self.params}


def execute_plan_steps(
    *,
    steps: list[PlanStep],
    registry: OperatorRegistry,
    task: TaskTemplate,
    experiment: Experiment,
    initial_artifacts: list[Artifact],
) -> None:
    current_artifacts = list(initial_artifacts)
    for step in steps:
        operator = registry.get(step.operator_name)
        input_ids = [artifact.artifact_id for artifact in current_artifacts]
        try:
            result = operator.run(
                current_artifacts,
                sandbox=experiment.sandbox,
                task=task,
                experiment=experiment,
                **step.params,
            )
            output_artifacts = result if isinstance(result, list) else [result]
            output_ids: list[str] = []
            next_artifacts: list[Artifact] = []
            for artifact in output_artifacts:
                artifact.save_meta()
                experiment.add_artifact(artifact)
                next_artifacts.append(artifact)
                output_ids.append(artifact.artifact_id)
            current_artifacts = next_artifacts
            experiment.record_step(
                operator_name=step.operator_name,
                input_artifact_ids=input_ids,
                output_artifact_ids=output_ids,
                params=step.params,
                status="completed",
            )
        except Exception as exc:
            experiment.record_step(
                operator_name=step.operator_name,
                input_artifact_ids=input_ids,
                output_artifact_ids=[],
                params=step.params,
                status="failed",
                error=str(exc),
            )
            experiment.fail(exc)
            raise


@dataclass(slots=True)
class LLMPlanningAgent:
    """Single-experiment planner and reviewer used by ``run_task``."""

    llm_client: LLMClient
    registry: OperatorRegistry
    max_steps: int = PLAN_MAX_STEPS
    last_plan_payload: dict[str, Any] | None = None
    last_review_payload: dict[str, Any] | None = None

    def plan(
        self,
        task: TaskTemplate,
        initial_artifact_summaries: list[dict[str, Any]],
        history_summaries: list[dict[str, Any]],
        operator_descriptions: list[dict[str, Any]] | None = None,
        prompt_rules: str | None = None,
    ) -> list[PlanStep]:
        del prompt_rules  # Reserved for future overrides.
        operator_descriptions = operator_descriptions or self.registry.describe_all()
        operator_lookup = {operator.name: operator for operator in self.registry.list()}
        system_prompt = build_plan_system_prompt(self.max_steps)
        user_prompt = build_plan_user_prompt(
            task.to_record(),
            initial_artifact_summaries,
            history_summaries,
            operator_descriptions,
        )
        try:
            payload = self.llm_client.generate_json(system_prompt, user_prompt, temperature=0.0)
            normalized = validate_plan_payload(payload, operator_lookup, max_steps=self.max_steps)
        except Exception as exc:
            raise RuntimeError(f"Agent planning failed: {exc}") from exc
        self.last_plan_payload = normalized
        return [PlanStep(**step) for step in normalized["steps"]]

    def run_experiment(
        self,
        task: TaskTemplate,
        experiment: Experiment,
        initial_artifacts: list[Artifact],
        history_summaries: list[dict[str, Any]] | None = None,
    ) -> list[PlanStep]:
        history_summaries = history_summaries or []
        initial_summaries = [artifact.summary() for artifact in initial_artifacts]
        try:
            steps = self.plan(
                task=task,
                initial_artifact_summaries=initial_summaries,
                history_summaries=history_summaries,
                operator_descriptions=self.registry.describe_all(),
            )
        except Exception as exc:
            experiment.fail(exc)
            raise
        if self.last_plan_payload is not None:
            experiment.set_plan(self.last_plan_payload)
            experiment.sandbox.write_json("plan.json", self.last_plan_payload)
        execute_plan_steps(
            steps=steps,
            registry=self.registry,
            task=task,
            experiment=experiment,
            initial_artifacts=initial_artifacts,
        )
        return steps

    def review(self, task: TaskTemplate, experiment: Experiment) -> dict[str, Any]:
        system_prompt = build_review_system_prompt()
        user_prompt = build_review_user_prompt(
            task.to_record(),
            experiment.to_record(),
            self.registry.describe_all(),
        )
        try:
            payload = self.llm_client.generate_json(system_prompt, user_prompt, temperature=0.0)
            normalized = normalize_review_payload(payload)
        except Exception as exc:
            normalized = {
                "summary": f"Fallback review generated because LLM output was invalid: {exc}",
                "issues": [] if experiment.status == "completed" else [experiment.error or "Experiment failed"],
                "next_hypothesis": "Tighten the plan or inspect the produced artifacts.",
                "recommended_changes": ["Review the experiment report before starting another round."],
            }
        self.last_review_payload = normalized
        experiment.set_review(normalized)
        experiment.sandbox.write_json("review.json", normalized)
        return normalized
