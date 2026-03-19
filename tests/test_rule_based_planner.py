import unittest

from dataos.core import (
    BudgetSpec,
    EvalSpec,
    IntelligenceLevel,
    OperatorContext,
    OperatorResult,
    OperatorSpec,
    TaskSpec,
    TaskType,
    TrainingStage,
)
from dataos.planner import PlannerRequest, PlanningError, RuleBasedPlanner
from dataos.registry import OperatorRegistry


class StubOperator:
    def __init__(self, operator_id: str) -> None:
        self.spec = OperatorSpec(
            operator_id=operator_id,
            name=operator_id,
            category="utility",
            input_type="records",
            output_type="records",
        )

    def execute(self, context: OperatorContext) -> OperatorResult:
        del context
        return OperatorResult()


class RuleBasedPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = RuleBasedPlanner()

    def test_clean_task_maps_to_filter_and_dedup(self) -> None:
        task = self._make_task(
            task_type=TaskType.CLEAN,
            allowed_operators=["filter_by_rule", "dedup"],
        )

        steps = self.planner.plan(task)

        self.assertEqual(
            [step.operator_id for step in steps],
            ["filter_by_rule", "dedup"],
        )
        self.assertEqual(steps[0].inputs, ["sft-mvp-task:input"])
        self.assertEqual(steps[1].outputs, ["sft-mvp-task:output"])
        self.assertEqual(steps[0].config["required_fields"], ["prompt", "response", "source"])
        self.assertEqual(
            steps[1].config["dedup_keys"],
            ["id", "prompt_response_normalized"],
        )

    def test_select_task_maps_to_score_filter_and_sampling(self) -> None:
        task = self._make_task(
            task_type=TaskType.SELECT,
            allowed_operators=["filter_by_score", "sample_by_strategy"],
        )

        steps = self.planner.plan(task)

        self.assertEqual(
            [step.operator_id for step in steps],
            ["filter_by_score", "sample_by_strategy"],
        )
        self.assertEqual(steps[0].config["score_field"], "score")
        self.assertEqual(steps[0].config["threshold"], 0.7)
        self.assertEqual(steps[1].config["strategy"], "weighted_by_source")
        self.assertEqual(steps[1].config["max_samples"], 1024)
        self.assertEqual(steps[1].config["random_seed"], 17)

    def test_iterate_task_builds_sft_mvp_runtime_plan_without_eval_step(self) -> None:
        task = self._make_task(
            task_type=TaskType.ITERATE,
            allowed_operators=[
                "filter_by_rule",
                "dedup",
                "filter_by_score",
                "sample_by_strategy",
            ],
        )

        steps = self.planner.plan(task)

        self.assertEqual(
            [step.operator_id for step in steps],
            [
                "filter_by_rule",
                "dedup",
                "filter_by_score",
                "sample_by_strategy",
            ],
        )
        self.assertEqual(steps[-1].outputs, ["sft-mvp-task:output"])

    def test_plan_result_surfaces_post_runtime_eval_handoff(self) -> None:
        task = self._make_task(
            task_type=TaskType.ITERATE,
            allowed_operators=[
                "filter_by_rule",
                "dedup",
                "filter_by_score",
                "sample_by_strategy",
            ],
        )

        result = self.planner.plan_result(PlannerRequest(task_spec=task), OperatorRegistry())

        self.assertEqual(
            [step.operator_id for step in result.pipeline],
            ["filter_by_rule", "dedup", "filter_by_score", "sample_by_strategy"],
        )
        self.assertTrue(result.rationale["runtime_only_pipeline"])
        self.assertEqual(result.rationale["evaluation"]["handoff"], "post_runtime")
        self.assertEqual(result.rationale["evaluation"]["evaluator_id"], "simple_sft_eval")
        self.assertEqual(
            result.rationale["evaluation"]["metrics"],
            ["eval.win_rate", "quality.pass_rate"],
        )

    def test_evaluate_task_maps_to_no_runtime_steps(self) -> None:
        task = self._make_task(
            task_type=TaskType.EVALUATE,
            allowed_operators=[],
        )

        steps = self.planner.plan(task, input_ref="candidate-dataset:v3")

        self.assertEqual(steps, [])

    def test_missing_required_operator_raises_planning_error(self) -> None:
        task = self._make_task(
            task_type=TaskType.ITERATE,
            allowed_operators=["filter_by_rule", "dedup", "filter_by_score"],
        )

        with self.assertRaisesRegex(PlanningError, "sample_by_strategy"):
            self.planner.plan(task)

    def test_plan_result_rejects_missing_registered_operator_when_registry_is_populated(self) -> None:
        task = self._make_task(
            task_type=TaskType.CLEAN,
            allowed_operators=["filter_by_rule", "dedup"],
        )
        registry = OperatorRegistry([StubOperator("filter_by_rule")])

        with self.assertRaisesRegex(PlanningError, "dedup"):
            self.planner.plan_result(
                PlannerRequest(task_spec=task),
                registry,
            )

    def _make_task(
        self,
        *,
        task_type: TaskType,
        allowed_operators: list[str],
    ) -> TaskSpec:
        return TaskSpec(
            task_id="sft-mvp-task",
            name="SFT MVP Task",
            training_stage=TrainingStage.SFT,
            intelligence_level=IntelligenceLevel.OPTIMIZE_WITH_EVAL,
            task_type=task_type,
            input_schema={
                "type": "jsonl",
                "sample_unit": "conversation",
                "fields": [
                    {"name": "id", "type": "string"},
                    {"name": "prompt", "type": "string"},
                    {"name": "response", "type": "string"},
                    {"name": "source", "type": "string"},
                    {"name": "score", "type": "float"},
                ],
            },
            output_schema={
                "type": "jsonl",
                "sample_unit": "conversation",
                "fields": [
                    {"name": "prompt", "type": "string"},
                    {"name": "response", "type": "string"},
                    {"name": "tags", "type": "list[string]"},
                ],
            },
            constraints={
                "required_fields": ["prompt", "response", "source"],
                "forbidden_content": ["pii", "empty_assistant_reply"],
                "quality_gates": {"min_turns": 2, "max_prompt_tokens": 4096},
                "score_threshold": 0.7,
                "sampling_strategy": "weighted_by_source",
                "sample_group_by": ["source"],
            },
            objective={
                "primary_metric": "eval.win_rate",
                "direction": "maximize",
            },
            budget=BudgetSpec(max_samples=1024),
            eval_spec=EvalSpec(
                evaluator_id="simple_sft_eval",
                metrics=["eval.win_rate", "quality.pass_rate"],
                dataset_split="validation",
                artifacts=["report_json"],
                success_criteria={"eval.win_rate": 0.75},
            ),
            allowed_operators=allowed_operators,
            metadata={"random_seed": 17},
        )
