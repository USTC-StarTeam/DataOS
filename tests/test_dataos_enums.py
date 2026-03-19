import unittest

from dataos.core import (
    IntelligenceLevel,
    OperatorCategory,
    StepStatus,
    TaskType,
    TrainingStage,
)
from dataos.core import enums


class DataOSEnumsTests(unittest.TestCase):
    def test_core_reexports_shared_enum_definitions(self) -> None:
        self.assertIs(TrainingStage, enums.TrainingStage)
        self.assertIs(IntelligenceLevel, enums.IntelligenceLevel)
        self.assertIs(TaskType, enums.TaskType)
        self.assertIs(OperatorCategory, enums.OperatorCategory)
        self.assertIs(StepStatus, enums.StepStatus)

    def test_training_stage_values_match_design_vocab(self) -> None:
        self.assertEqual(
            [stage.value for stage in TrainingStage],
            ["pretrain", "continual_pretrain", "sft", "rl", "eval"],
        )

    def test_task_and_operator_vocab_are_distinct_but_aligned(self) -> None:
        self.assertEqual(
            [task_type.value for task_type in TaskType],
            ["clean", "select", "mix", "synthesize", "rewrite", "evaluate", "iterate"],
        )
        self.assertEqual(
            [category.value for category in OperatorCategory],
            [
                "cleaning",
                "selection",
                "scoring",
                "mixing",
                "synthesis",
                "rewriting",
                "evaluation",
                "utility",
            ],
        )

    def test_step_status_values_are_stable_for_runtime_lifecycle(self) -> None:
        self.assertEqual(
            [status.value for status in StepStatus],
            ["pending", "running", "succeeded", "failed", "skipped"],
        )
