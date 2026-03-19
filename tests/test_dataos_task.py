import unittest

from dataos.core import (
    BudgetSpec,
    EvalSpec,
    IntelligenceLevel,
    StopCondition,
    TaskSpec,
    TaskType,
    TrainingStage,
)


class TaskSpecTests(unittest.TestCase):
    def test_task_spec_can_express_iterative_sft_data_task(self) -> None:
        task = TaskSpec(
            task_id="sft-math-iter-v1",
            name="Math Reasoning SFT Data Iteration",
            training_stage=TrainingStage.SFT,
            intelligence_level=IntelligenceLevel.OPTIMIZE_WITH_EVAL,
            task_type=TaskType.ITERATE,
            input_schema={
                "type": "jsonl",
                "sample_unit": "conversation",
                "fields": [
                    {"name": "messages", "type": "list[chat_message]"},
                    {"name": "source", "type": "string"},
                    {"name": "quality_score", "type": "float"},
                ],
            },
            output_schema={
                "type": "jsonl",
                "sample_unit": "conversation",
                "fields": [
                    {"name": "messages", "type": "list[chat_message]"},
                    {"name": "tags", "type": "list[string]"},
                ],
            },
            constraints={
                "required_fields": ["messages", "source"],
                "forbidden_content": ["pii", "empty_assistant_reply"],
                "quality_gates": {"min_turns": 2, "max_prompt_tokens": 4096},
            },
            objective={
                "primary_metric": "eval.win_rate",
                "direction": "maximize",
                "secondary_metrics": ["format.valid_rate", "quality.pass_rate"],
            },
            budget=BudgetSpec(
                max_iterations=3,
                max_samples=50_000,
                max_model_calls=12,
                max_runtime_seconds=3_600,
                max_cost_usd=25.0,
            ),
            stop_condition=StopCondition(
                metric_thresholds={"eval.win_rate": 0.75, "format.valid_rate": 0.99},
                max_rounds_without_improvement=2,
                min_improvement_delta=0.01,
                custom_rules=["stop when no new high-quality sources are added"],
            ),
            eval_spec=EvalSpec(
                evaluator_id="simple_sft_eval",
                metrics=["eval.win_rate", "format.valid_rate", "quality.pass_rate"],
                dataset_split="validation",
                artifacts=["report_json", "failure_cases"],
                success_criteria={"format.valid_rate": 0.99},
            ),
            allowed_operators=[
                "filter_by_rule",
                "dedup",
                "rewrite_with_llm",
                "mix_datasets",
                "dedup",
            ],
            metadata={"owner": "dataos", "domain": "math_reasoning"},
        )

        payload = task.to_dict()

        self.assertEqual(payload["training_stage"], "sft")
        self.assertEqual(payload["intelligence_level"], "optimize_with_eval")
        self.assertEqual(payload["task_type"], "iterate")
        self.assertEqual(payload["budget"]["max_iterations"], 3)
        self.assertEqual(payload["eval_spec"]["evaluator_id"], "simple_sft_eval")
        self.assertEqual(
            task.allowed_operators,
            ["filter_by_rule", "dedup", "rewrite_with_llm", "mix_datasets"],
        )

    def test_task_spec_from_dict_round_trips_with_nested_specs(self) -> None:
        payload = {
            "task_id": "rewrite-v2",
            "name": "Rewrite low-quality SFT samples",
            "training_stage": "sft",
            "intelligence_level": "process",
            "task_type": "rewrite",
            "input_schema": {"type": "jsonl", "fields": [{"name": "messages"}]},
            "output_schema": {"type": "jsonl", "fields": [{"name": "messages"}]},
            "constraints": {"rewrite_policy": "preserve semantics"},
            "objective": {"primary_metric": "quality.score", "direction": "maximize"},
            "budget": {"max_iterations": 1, "max_cost_usd": 5.0},
            "stop_condition": {
                "metric_thresholds": {"quality.score": 0.9},
                "stop_on_budget_exhaustion": True,
            },
            "eval_spec": {
                "evaluator_id": "simple_sft_eval",
                "metrics": ["quality.score"],
                "dataset_split": "validation",
            },
            "allowed_operators": ["rewrite_with_llm"],
            "metadata": {"team": "foundation"},
        }

        task = TaskSpec.from_dict(payload)

        self.assertEqual(task.training_stage, TrainingStage.SFT)
        self.assertEqual(task.intelligence_level, IntelligenceLevel.PROCESS)
        self.assertEqual(task.task_type, TaskType.REWRITE)
        self.assertIsInstance(task.budget, BudgetSpec)
        self.assertIsInstance(task.stop_condition, StopCondition)
        self.assertIsInstance(task.eval_spec, EvalSpec)
        self.assertEqual(task.to_dict()["task_id"], payload["task_id"])
        self.assertEqual(task.to_dict()["budget"]["max_iterations"], 1)
        self.assertEqual(task.to_dict()["budget"]["max_cost_usd"], 5.0)
        self.assertEqual(task.to_dict()["eval_spec"]["metrics"], ["quality.score"])
        self.assertEqual(task.to_dict()["allowed_operators"], ["rewrite_with_llm"])

    def test_task_spec_validates_required_fields(self) -> None:
        with self.assertRaises(ValueError):
            TaskSpec(
                task_id="",
                name="Invalid",
                training_stage=TrainingStage.SFT,
                intelligence_level=IntelligenceLevel.PROCESS,
                task_type=TaskType.CLEAN,
                input_schema={"type": "jsonl"},
                output_schema={"type": "jsonl"},
                eval_spec={"evaluator_id": "simple_sft_eval"},
            )

        with self.assertRaises(ValueError):
            BudgetSpec(max_iterations=-1)

        with self.assertRaises(ValueError):
            EvalSpec(evaluator_id=" ", metrics=["quality.score"])
