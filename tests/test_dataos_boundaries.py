from __future__ import annotations

import unittest

from dataos.controller import Controller, ControllerDecision, ControllerRoundContext
from dataos.core import (
    BudgetSpec,
    DatasetVersion,
    EvalResult,
    EvalSpec,
    IntelligenceLevel,
    OptimizationState,
    PipelineStep,
    StepStatus,
    TaskSpec,
    TaskType,
    TrainingStage,
)
from dataos.eval import AnalysisSummary, Evaluator, SimpleSFTEvaluator
from dataos.planner import Planner, PlannerRequest, PlanningResult, RuleBasedPlanner
from dataos.registry import OperatorRegistry
from dataos.runtime import RuntimeExecutionRequest, RuntimeExecutionResult, RuntimeStateStore
from dataos.storage import RunSnapshot


class StaticController:
    controller_id = "static"

    def decide(self, context: ControllerRoundContext) -> ControllerDecision:
        return ControllerDecision(
            should_continue=False,
            next_action="stop",
            selected_plan=context.latest_plan,
            updated_state=context.optimization_state,
            best_result=context.latest_evals[-1] if context.latest_evals else None,
            reason="boundary-smoke-test",
        )


class InMemoryStateStore:
    def __init__(self) -> None:
        self._snapshots: dict[str, RunSnapshot] = {}

    def save(self, snapshot: RunSnapshot) -> None:
        self._snapshots[snapshot.run_id] = snapshot

    def load(self, run_id: str) -> RunSnapshot | None:
        return self._snapshots.get(run_id)


class DataOSBoundaryTests(unittest.TestCase):
    def test_rule_based_planner_emits_planning_result_boundary_object(self) -> None:
        task = self._make_task()
        dataset = DatasetVersion(
            dataset_id="sft-input",
            version="v1",
            source={"kind": "seed"},
        )
        planner = RuleBasedPlanner()

        result = planner.plan_result(
            PlannerRequest(task_spec=task, datasets=[dataset]),
            OperatorRegistry(),
        )

        self.assertEqual(result.plan_id, "boundary-task:rule_based")
        self.assertEqual(result.planner_id, "rule_based")
        self.assertEqual(result.pipeline[0].inputs, ["sft-input:v1"])
        self.assertEqual(result.pipeline[-1].outputs, ["boundary-task:output"])
        self.assertTrue(result.rationale["runtime_only_pipeline"])
        self.assertEqual(result.rationale["evaluation"]["handoff"], "post_runtime")
        self.assertEqual(result.rationale["evaluation"]["evaluator_id"], "simple_sft_eval")

    def test_boundary_protocols_keep_planning_runtime_and_control_separate(self) -> None:
        task = self._make_task()
        step = PipelineStep(
            step_id="clean-01-filter_by_rule",
            operator_id="filter_by_rule",
            inputs=["sft-input:v1"],
            outputs=["boundary-task:output"],
        )
        plan = PlanningResult(
            plan_id="boundary-task:rule_based",
            planner_id="rule_based",
            pipeline=[step],
            candidate_config={"strategy": "rule_based"},
        )
        runtime_request = RuntimeExecutionRequest(
            pipeline=[step],
            initial_inputs={"sft-input:v1": [{"prompt": "hi", "response": "hello"}]},
            run_id="run-001",
            task_spec=task,
        )
        runtime_result = RuntimeExecutionResult(
            run_id="run-001",
            status=StepStatus.SUCCEEDED,
            steps=[step],
            metrics={"planned_steps": 1, "completed_steps": 1},
        )
        eval_result = EvalResult(
            eval_id="eval-001",
            dataset_version="boundary-task:output",
            task_id=task.task_id,
            metrics={"eval.win_rate": 0.8},
        )
        summary = AnalysisSummary(
            summary_id="summary-001",
            winner_eval_id=eval_result.eval_id,
            winner_dataset_version=eval_result.dataset_version,
            metric_deltas={"eval.win_rate": 0.05},
        )
        state = OptimizationState(
            round_id="round-001",
            best_result=eval_result,
        )
        context = ControllerRoundContext(
            task_spec=task,
            optimization_state=state,
            latest_plan=plan,
            latest_runtime=runtime_result,
            latest_evals=[eval_result],
            analysis=summary,
        )
        controller = StaticController()
        store = InMemoryStateStore()

        self.assertIsInstance(RuleBasedPlanner(), Planner)
        self.assertIsInstance(controller, Controller)
        self.assertIsInstance(store, RuntimeStateStore)

        decision = controller.decide(context)
        store.save(
            RunSnapshot(
                run_id=runtime_request.run_id or "run-001",
                plan=plan.pipeline,
                steps=runtime_result.steps,
                eval_results=[eval_result],
                optimization_state=state,
            )
        )

        self.assertFalse(decision.should_continue)
        self.assertEqual(decision.next_action, "stop")
        self.assertEqual(decision.selected_plan.plan_id, "boundary-task:rule_based")
        self.assertEqual(store.load("run-001").run_id, "run-001")

    def test_simple_sft_eval_is_an_evaluator_boundary(self) -> None:
        evaluator = SimpleSFTEvaluator()

        self.assertIsInstance(evaluator, Evaluator)
        self.assertEqual(evaluator.spec.evaluator_id, "simple_sft_eval")

    def _make_task(self) -> TaskSpec:
        return TaskSpec(
            task_id="boundary-task",
            name="Boundary Task",
            training_stage=TrainingStage.SFT,
            intelligence_level=IntelligenceLevel.OPTIMIZE_WITH_EVAL,
            task_type=TaskType.CLEAN,
            input_schema={
                "type": "jsonl",
                "fields": [
                    {"name": "prompt", "type": "string"},
                    {"name": "response", "type": "string"},
                ],
            },
            output_schema={
                "type": "jsonl",
                "fields": [
                    {"name": "prompt", "type": "string"},
                    {"name": "response", "type": "string"},
                ],
            },
            constraints={"required_fields": ["prompt", "response"]},
            objective={"primary_metric": "eval.win_rate", "direction": "maximize"},
            budget=BudgetSpec(max_iterations=3),
            eval_spec=EvalSpec(
                evaluator_id="simple_sft_eval",
                metrics=["eval.win_rate"],
                dataset_split="validation",
            ),
            allowed_operators=["filter_by_rule", "dedup"],
        )
