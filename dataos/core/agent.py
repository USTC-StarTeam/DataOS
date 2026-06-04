"""Plan/review boundary for ``run_task``, powered by the external ``dataos_agent``.

This module no longer hand-rolls prompts + HTTP + JSON parsing. Instead it drives
:class:`dataos_agent.Agent` and asks for *structured* output (a Pydantic model via
``output=``), which the agent backend validates and returns as a typed object.

Two backends matter here:

* the ``openai`` backend talks to any OpenAI-compatible endpoint (MiMo by default),
  configured entirely from environment variables (no hardcoded keys);
* the ``fake`` backend is fully offline (no SDK, no network) and is what the tests
  use to drive plan + review deterministically.

The public surface used by the runtime (``LLMPlanningAgent``, ``PlanStep``,
``execute_plan_steps``) is unchanged so the rest of DataOS keeps working.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from dataos.core.artifact import Artifact
from dataos.core.experiment import Experiment
from dataos.core.registry import OperatorRegistry
from dataos.core.task import TaskTemplate

# ``dataos_agent`` is now the single LLM boundary for plan/review.
from dataos_agent import Agent
from dataos_agent.config import AgentConfig

PLAN_MAX_STEPS = 6

# Default OpenAI-compatible endpoint + model for a live MiMo run. Overridable via env.
DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_MODEL = "mimo-v2.5-pro"


# --------------------------------------------------------------------------- #
# Structured output schemas (what the agent must return).
# --------------------------------------------------------------------------- #
class PlanStepModel(BaseModel):
    """One planned operator invocation."""

    operator_name: str
    params: dict[str, Any] = Field(default_factory=dict)


class PlanModel(BaseModel):
    """Structured PLAN returned by the agent."""

    steps: list[PlanStepModel] = Field(default_factory=list)
    reasoning_summary: str = ""
    expected_gain: str = ""


class ReviewModel(BaseModel):
    """Structured REVIEW returned by the agent."""

    summary: str = ""
    issues: list[str] = Field(default_factory=list)
    next_hypothesis: str = ""
    recommended_changes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Agent construction (env-driven, provider selectable for tests).
# --------------------------------------------------------------------------- #
def build_agent_from_env(
    *,
    provider: Optional[str] = None,
    workspace: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Agent:
    """Build a :class:`dataos_agent.Agent` for the plan/review boundary.

    Provider resolution order: explicit ``provider=`` arg, then ``DATAOS_AGENT_PROVIDER``
    env, then ``openai``. The special value ``fake`` selects the offline test backend
    (no key, no network). For the ``openai`` backend the key comes from ``MIMO_API_KEY``
    or ``OPENAI_API_KEY``, base url from ``OPENAI_BASE_URL`` (default MiMo), and model
    from ``DATAOS_AGENT_MODEL`` / ``OPENAI_MODEL`` (default ``mimo-v2.5-pro``). No API key
    is ever hardcoded.
    """
    provider = (provider or os.environ.get("DATAOS_AGENT_PROVIDER") or "openai").strip().lower()
    model = os.environ.get("DATAOS_AGENT_MODEL") or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL

    if provider == "fake":
        # Offline backend: no SDK, no network. Construct the config directly because
        # Agent.from_config()/factories only cover the real backends. The fake backend
        # echoes config.extra["reply"] as the model output for every run, so an env
        # var lets an offline end-to-end run inject a structured plan/review reply.
        fake_extra = dict(extra or {})
        env_reply = os.environ.get("DATAOS_AGENT_FAKE_REPLY")
        if env_reply and "reply" not in fake_extra:
            fake_extra["reply"] = env_reply
        cfg = AgentConfig(
            provider="fake",
            model=model,
            workspace=workspace,
            extra=fake_extra,
        )
        return Agent(cfg)

    if provider in {"openai", "openai_agents"}:
        api_key = os.environ.get("MIMO_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing LLM API key for the dataos_agent openai backend. "
                "Set MIMO_API_KEY (or OPENAI_API_KEY). "
                "Optional: OPENAI_BASE_URL, DATAOS_AGENT_MODEL, DATAOS_AGENT_PROVIDER."
            )
        base_url = os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        return Agent.openai(
            model=model,
            api_key=api_key,
            base_url=base_url,
            workspace=workspace,
            extra=dict(extra or {}),
        )

    raise ValueError(
        f"Unsupported DATAOS_AGENT_PROVIDER {provider!r}. Use 'openai' or 'fake'."
    )


# --------------------------------------------------------------------------- #
# Prompt building (plain context strings; the schema is enforced by ``output=``).
# --------------------------------------------------------------------------- #
_PLAN_RULES = [
    "Only choose operators that appear in available_operators.",
    f"Use no more than {PLAN_MAX_STEPS} steps.",
    "Only plan data operators. Training and evaluation run automatically as fixed runtime stages afterwards.",
    "Do not invent new operators, modify code, or run shell commands.",
    "Treat all datasets as Hugging Face dataset artifacts, never JSONL.",
    "Use only parameters that appear in an operator's schema; prefer defaults when the task does not constrain them.",
    "Operators always consume the latest upstream dataset artifact, so order the steps to be executable in sequence.",
    "If the latest dataset lacks Alpaca-style columns (instruction/input/output), use map_to_alpaca before filters or evaluators that rely on them.",
]

_REVIEW_RULES = [
    "Summarize the experiment, surface issues, and propose the next hypothesis.",
    "Do not request code changes or shell commands.",
]


def build_plan_prompt(
    task_record: dict[str, Any],
    initial_artifacts: list[dict[str, Any]],
    history: list[dict[str, Any]],
    operators: list[dict[str, Any]],
) -> str:
    payload = {
        "task": task_record,
        "initial_artifacts": initial_artifacts,
        "history": history,
        "available_operators": operators,
        "execution_note": (
            "Each operator runs against the latest upstream dataset artifact, "
            "which is initially the last dataset in initial_artifacts."
        ),
    }
    rules = "\n".join(f"- {rule}" for rule in _PLAN_RULES)
    return (
        "You are the planning component for DataOS. Produce an ordered list of data "
        "operator steps to improve the SFT dataset for the given task.\n"
        f"The runtime can execute at most {PLAN_MAX_STEPS} operators.\n"
        "Rules:\n"
        f"{rules}\n\n"
        "Context (JSON):\n"
        f"{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}"
    )


def build_review_prompt(
    task_record: dict[str, Any],
    experiment_record: dict[str, Any],
    operators: list[dict[str, Any]],
) -> str:
    payload = {
        "task": task_record,
        "experiment": experiment_record,
        "available_operators": operators,
    }
    rules = "\n".join(f"- {rule}" for rule in _REVIEW_RULES)
    return (
        "You are the review component for DataOS. Review the finished experiment.\n"
        "Rules:\n"
        f"{rules}\n\n"
        "Context (JSON):\n"
        f"{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}"
    )


# --------------------------------------------------------------------------- #
# Plan steps + execution (unchanged contract).
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Planner/reviewer driven by dataos_agent.Agent.
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class LLMPlanningAgent:
    """Single-experiment planner and reviewer used by ``run_task``.

    Plan and review now flow through a :class:`dataos_agent.Agent` with structured
    (Pydantic) output. ``agent`` is injectable so tests can pass a ``fake``-backed
    agent; if omitted it is built from the environment.
    """

    agent: Agent
    registry: OperatorRegistry
    max_steps: int = PLAN_MAX_STEPS
    last_plan_payload: dict[str, Any] | None = None
    last_review_payload: dict[str, Any] | None = None

    def _resolve_steps(self, plan: PlanModel) -> tuple[list[dict[str, Any]], list[PlanStep]]:
        """Validate operator names against the registry and normalize params.

        Unknown operators / bad params are dropped (best-effort). Falls back to the
        first registered operator if nothing valid survives, so the pipeline always runs.
        """
        operator_lookup = {operator.name: operator for operator in self.registry.list()}
        normalized_steps: list[dict[str, Any]] = []
        for step in plan.steps[: self.max_steps]:
            operator = operator_lookup.get(step.operator_name)
            if operator is None:
                continue
            try:
                params = operator.normalize_params(dict(step.params or {}))
            except Exception:
                continue
            normalized_steps.append({"operator_name": step.operator_name, "params": params})

        reasoning = plan.reasoning_summary
        expected_gain = plan.expected_gain
        if not normalized_steps and operator_lookup:
            fallback_name = sorted(operator_lookup)[0]
            normalized_steps.append(
                {"operator_name": fallback_name, "params": operator_lookup[fallback_name].default_params()}
            )
            reasoning = reasoning or f"Fallback plan: agent returned no valid steps; ran {fallback_name}."
            expected_gain = expected_gain or "Establish a baseline pipeline run."

        payload = {
            "steps": normalized_steps,
            "reasoning_summary": str(reasoning),
            "expected_gain": str(expected_gain),
        }
        return payload, [PlanStep(**step) for step in normalized_steps]

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
        prompt = build_plan_prompt(
            task.to_record(),
            initial_artifact_summaries,
            history_summaries,
            operator_descriptions,
        )
        try:
            result = self.agent.run(prompt, output=PlanModel)
            plan_model = result.output
            if not isinstance(plan_model, PlanModel):
                plan_model = PlanModel()
        except Exception:
            # Network/parse failure: keep the pipeline alive with an empty plan that
            # _resolve_steps turns into a safe single-operator fallback.
            plan_model = PlanModel()
        payload, steps = self._resolve_steps(plan_model)
        self.last_plan_payload = payload
        return steps

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
        prompt = build_review_prompt(
            task.to_record(),
            experiment.to_record(),
            self.registry.describe_all(),
        )
        try:
            result = self.agent.run(prompt, output=ReviewModel)
            review_model = result.output
            if not isinstance(review_model, ReviewModel):
                raise ValueError("agent returned no structured review")
            normalized = {
                "summary": str(review_model.summary),
                "issues": [str(item) for item in review_model.issues],
                "next_hypothesis": str(review_model.next_hypothesis),
                "recommended_changes": [str(item) for item in review_model.recommended_changes],
            }
        except Exception as exc:
            normalized = {
                "summary": f"Fallback review generated because agent output was invalid: {exc}",
                "issues": [] if experiment.status == "completed" else [experiment.error or "Experiment failed"],
                "next_hypothesis": "Tighten the plan or inspect the produced artifacts.",
                "recommended_changes": ["Review the experiment report before starting another round."],
            }
        self.last_review_payload = normalized
        experiment.set_review(normalized)
        experiment.sandbox.write_json("review.json", normalized)
        return normalized
