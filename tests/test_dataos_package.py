import importlib
import unittest
from dataclasses import asdict

import dataos
import py_agent
from dataos.core import EvalResult, OptimizationState, PipelineStep, PipelineStepStatus


class DataOSPackageTests(unittest.TestCase):
    def test_dataos_package_imports_without_affecting_py_agent(self) -> None:
        self.assertEqual(dataos.__version__, "0.1.0")
        self.assertEqual(py_agent.__version__, "0.1.0")

    def test_dataos_subpackages_are_importable(self) -> None:
        module_names = [
            "dataos.controller",
            "dataos.controller.base",
            "dataos.controller.iterative_controller",
            "dataos.core",
            "dataos.core.eval",
            "dataos.core.optimization",
            "dataos.core.pipeline",
            "dataos.eval",
            "dataos.eval.base",
            "dataos.eval.offline_proxy_eval",
            "dataos.eval.result_analyzer",
            "dataos.eval.simple_sft_eval",
            "dataos.experiments",
            "dataos.experiments.sft",
            "dataos.llm",
            "dataos.llm.adapters",
            "dataos.operators",
            "dataos.operators.cleaning",
            "dataos.operators.evaluation",
            "dataos.operators.mixing",
            "dataos.operators.rewriting",
            "dataos.operators.selection",
            "dataos.operators.synthesis",
            "dataos.operators.utility",
            "dataos.planner",
            "dataos.planner.base",
            "dataos.registry",
            "dataos.registry.evaluator_registry",
            "dataos.registry.operator_registry",
            "dataos.registry.strategy_registry",
            "dataos.runtime",
            "dataos.runtime.checkpoint",
            "dataos.runtime.executor",
            "dataos.runtime.retry",
            "dataos.runtime.state_store",
            "dataos.storage",
            "dataos.storage.artifact_store",
            "dataos.storage.dataset_store",
            "dataos.storage.lineage_store",
            "dataos.storage.metrics_store",
            "dataos.storage.run_store",
        ]

        for module_name in module_names:
            with self.subTest(module_name=module_name):
                importlib.import_module(module_name)

    def test_pipeline_step_structures_runtime_state(self) -> None:
        step = PipelineStep(
            step_id="clean-1",
            operator_id="cleaning.normalize_text",
            inputs=["dataset:v1"],
            config={"lowercase": True},
            outputs=["dataset:v2"],
            status=PipelineStepStatus.SUCCEEDED,
            metrics={"rows_in": 100, "rows_out": 96},
        )

        self.assertEqual(
            step.to_dict(),
            {
                "step_id": "clean-1",
                "operator_id": "cleaning.normalize_text",
                "inputs": ["dataset:v1"],
                "config": {"lowercase": True},
                "outputs": ["dataset:v2"],
                "status": "succeeded",
                "metrics": {"rows_in": 100, "rows_out": 96},
            },
        )
        self.assertEqual(asdict(step)["status"], PipelineStepStatus.SUCCEEDED)

    def test_eval_and_optimization_state_are_nested_and_serializable(self) -> None:
        eval_result = EvalResult(
            eval_id="eval-1",
            dataset_version="dataset:v2",
            task_id="task-sft",
            metrics={"accuracy": 0.91, "loss": 0.12},
            artifacts={"report": "artifacts/eval-1.json"},
            summary={"winner": "candidate-a"},
        )
        optimization_state = OptimizationState(
            round_id="round-1",
            candidate_configs=[{"name": "candidate-a"}, {"name": "candidate-b"}],
            tried_configs=[{"name": "candidate-a"}],
            best_config={"name": "candidate-a"},
            best_result=eval_result,
            history=[
                {
                    "candidate": "candidate-a",
                    "eval_id": "eval-1",
                    "metrics": {"accuracy": 0.91},
                }
            ],
        )

        self.assertEqual(eval_result.to_dict()["metrics"]["accuracy"], 0.91)
        self.assertEqual(
            optimization_state.to_dict(),
            {
                "round_id": "round-1",
                "candidate_configs": [{"name": "candidate-a"}, {"name": "candidate-b"}],
                "tried_configs": [{"name": "candidate-a"}],
                "best_config": {"name": "candidate-a"},
                "best_result": {
                    "eval_id": "eval-1",
                    "dataset_version": "dataset:v2",
                    "task_id": "task-sft",
                    "metrics": {"accuracy": 0.91, "loss": 0.12},
                    "artifacts": {"report": "artifacts/eval-1.json"},
                    "summary": {"winner": "candidate-a"},
                },
                "history": [
                    {
                        "candidate": "candidate-a",
                        "eval_id": "eval-1",
                        "metrics": {"accuracy": 0.91},
                    }
                ],
            },
        )
