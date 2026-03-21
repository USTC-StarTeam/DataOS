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


@dataclass(slots=True)
class LLMPlanningAgent:
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
        operator_descriptions: list[dict[str, str]] | None = None,
        prompt_rules: str | None = None,
    ) -> list[PlanStep]:
        del prompt_rules  # Reserved for future overrides.
        operator_descriptions = operator_descriptions or self.registry.describe_all()
        operator_names = {item["name"] for item in operator_descriptions}
        system_prompt = build_plan_system_prompt(self.max_steps)
        user_prompt = build_plan_user_prompt(
            task.to_record(),
            initial_artifact_summaries,
            history_summaries,
            operator_descriptions,
        )
        try:
            payload = self.llm_client.generate_json(system_prompt, user_prompt, temperature=0.0)
            normalized = validate_plan_payload(payload, operator_names, max_steps=self.max_steps)
            self.last_plan_payload = normalized
            return [PlanStep(**step) for step in normalized["steps"]]
        except Exception as exc:
            fallback = self._fallback_plan()
            self.last_plan_payload = {
                "steps": [step.to_record() for step in fallback],
                "reasoning_summary": f"Fallback plan used because LLM output was invalid: {exc}",
                "expected_gain": "Run the default clean → synthesize → train → eval pipeline safely.",
            }
            return fallback

    def run_experiment(
        self,
        task: TaskTemplate,
        experiment: Experiment,
        initial_artifacts: list[Artifact],
        history_summaries: list[dict[str, Any]] | None = None,
    ) -> list[PlanStep]:
        history_summaries = history_summaries or []
        experiment.start()
        current_artifacts = list(initial_artifacts)
        initial_summaries = [artifact.summary() for artifact in initial_artifacts]
        steps = self.plan(
            task=task,
            initial_artifact_summaries=initial_summaries,
            history_summaries=history_summaries,
            operator_descriptions=self.registry.describe_all(),
        )
        if self.last_plan_payload is not None:
            experiment.sandbox.write_json("plan.json", self.last_plan_payload)

        for step in steps:
            operator = self.registry.get(step.operator_name)
            input_ids = [artifact.artifact_id for artifact in current_artifacts]
            try:
                result = operator.run(current_artifacts, sandbox=experiment.sandbox, task=task, experiment=experiment, **step.params)
                output_artifacts = result if isinstance(result, list) else [result]
                output_ids: list[str] = []
                for artifact in output_artifacts:
                    artifact.save_meta()
                    experiment.add_artifact(artifact)
                    current_artifacts.append(artifact)
                    output_ids.append(artifact.artifact_id)
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
        experiment.sandbox.write_json("review.json", normalized)
        return normalized

    def _fallback_plan(self) -> list[PlanStep]:
        preferred = [
            "clean_data",
            "synthesize_data",
            "train_llamafactory",
            "eval_model",
        ]
        available = {operator.name for operator in self.registry.list()}
        steps = [PlanStep(name, {}) for name in preferred if name in available]
        if len(steps) < 2 or steps[-2].operator_name != "train_llamafactory" or steps[-1].operator_name != "eval_model":
            raise ValueError("Registry does not contain the required fallback operators")
        return steps
