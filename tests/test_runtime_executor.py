from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from dataos.core import OperatorContext, OperatorResult, OperatorSpec, PipelineStep, StepStatus
from dataos.runtime import PipelineExecutionError, RunStoreLike, RuntimeExecutor, RuntimeStateStore
from dataos.storage import RunStore


class PrefixOperator:
    spec = OperatorSpec(
        operator_id="utility.prefix",
        name="Prefix",
        category="utility",
        input_type="records",
        output_type="records",
    )

    def execute(self, context: OperatorContext) -> OperatorResult:
        prefix = context.config.get("prefix", "")
        return OperatorResult(
            outputs=[f"{prefix}{item}" for item in context.inputs],
            metrics={"input_count": len(context.inputs), "output_count": len(context.inputs)},
            metadata={"observed_resources": dict(context.resources)},
        )


class SuffixOperator:
    spec = OperatorSpec(
        operator_id="utility.suffix",
        name="Suffix",
        category="utility",
        input_type="records",
        output_type="records",
    )

    def execute(self, context: OperatorContext) -> OperatorResult:
        suffix = context.config.get("suffix", "")
        return OperatorResult(
            outputs=[f"{item}{suffix}" for item in context.inputs],
            metrics={"input_count": len(context.inputs), "output_count": len(context.inputs)},
        )


class FailingOperator:
    spec = OperatorSpec(
        operator_id="utility.fail",
        name="Fail",
        category="utility",
        input_type="records",
        output_type="records",
    )

    def execute(self, context: OperatorContext) -> OperatorResult:
        raise RuntimeError(f"cannot process {context.step_id}")


@dataclass
class InMemoryRunStore:
    saved_plans: list[dict[str, Any]] = field(default_factory=list)
    saved_steps: list[dict[str, Any]] = field(default_factory=list)
    saved_results: list[dict[str, Any]] = field(default_factory=list)

    def save_plan(
        self,
        run_id: str,
        pipeline: list[PipelineStep],
        task_spec: Any | None = None,
    ) -> None:
        self.saved_plans.append(
            {
                "run_id": run_id,
                "step_ids": [step.step_id for step in pipeline],
                "task_id": getattr(task_spec, "task_id", None),
            }
        )

    def save_step(self, run_id: str, step: PipelineStep, record: Any) -> None:
        self.saved_steps.append(
            {
                "run_id": run_id,
                "step_id": step.step_id,
                "status": step.status,
                "record": record,
            }
        )

    def save_result(self, run_id: str, result: Any) -> None:
        self.saved_results.append({"run_id": run_id, "result": result})


class RuntimeExecutorTests(unittest.TestCase):
    def test_executor_can_persist_and_reload_run_snapshot_with_run_store(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = RunStore(Path(tmpdir) / "artifacts" / "runs")
            executor = RuntimeExecutor(
                {
                    "utility.prefix": PrefixOperator(),
                    "utility.suffix": SuffixOperator(),
                },
                run_store=store,
            )

            result = executor.execute(
                [
                    PipelineStep(
                        step_id="prefix-step",
                        operator_id="utility.prefix",
                        inputs=["seed"],
                        outputs=["prefixed"],
                        config={"prefix": "x-"},
                    ),
                    PipelineStep(
                        step_id="suffix-step",
                        operator_id="utility.suffix",
                        inputs=["prefixed"],
                        outputs=["final"],
                        config={"suffix": "-z"},
                    ),
                ],
                initial_inputs={"seed": ["a", "b"]},
                run_id="run-store-1",
            )

            snapshot = store.load("run-store-1")

            self.assertEqual(result.status, StepStatus.SUCCEEDED)
            self.assertIsInstance(store, RunStoreLike)
            self.assertIsInstance(store, RuntimeStateStore)
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual([step.step_id for step in snapshot.plan], ["prefix-step", "suffix-step"])
            self.assertEqual(
                [step.status for step in snapshot.steps],
                [StepStatus.SUCCEEDED, StepStatus.SUCCEEDED],
            )
            self.assertEqual(snapshot.metrics["planned_steps"], 2)
            self.assertEqual(snapshot.metrics["succeeded_steps"], 2)
            self.assertEqual(snapshot.metadata["runtime"]["status"], StepStatus.SUCCEEDED.value)
            self.assertEqual(snapshot.metadata["runtime"]["state_keys"], ["final", "prefixed", "seed"])

            prefix_record = json.loads(
                (Path(tmpdir) / "artifacts" / "runs" / "run-store-1" / "records" / "prefix-step.json")
                .read_text(encoding="utf-8")
            )
            suffix_record = json.loads(
                (Path(tmpdir) / "artifacts" / "runs" / "run-store-1" / "records" / "suffix-step.json")
                .read_text(encoding="utf-8")
            )

            self.assertEqual(prefix_record["inputs"], ["a", "b"])
            self.assertEqual(prefix_record["outputs"], ["x-a", "x-b"])
            self.assertEqual(prefix_record["config"], {"prefix": "x-"})
            self.assertEqual(prefix_record["metrics"], {"input_count": 2, "output_count": 2})
            self.assertGreaterEqual(prefix_record["duration_seconds"], 0.0)
            self.assertEqual(suffix_record["inputs"], ["x-a", "x-b"])
            self.assertEqual(suffix_record["outputs"], ["x-a-z", "x-b-z"])

    def test_executor_runs_multi_step_pipeline_and_records_observability(self) -> None:
        run_store = InMemoryRunStore()
        executor = RuntimeExecutor(
            {
                "utility.prefix": PrefixOperator(),
                "utility.suffix": SuffixOperator(),
            },
            run_store=run_store,
            resources={"cluster": "local"},
        )

        result = executor.execute(
            [
                PipelineStep(
                    step_id="prefix-step",
                    operator_id="utility.prefix",
                    inputs=["seed"],
                    outputs=["prefixed"],
                    config={"prefix": "x-"},
                ),
                PipelineStep(
                    step_id="suffix-step",
                    operator_id="utility.suffix",
                    inputs=["prefixed"],
                    outputs=["final"],
                    config={"suffix": "-z"},
                ),
            ],
            initial_inputs={"seed": ["a", "b"]},
            run_id="run-123",
        )

        self.assertEqual(result.run_id, "run-123")
        self.assertEqual(result.status, StepStatus.SUCCEEDED)
        self.assertEqual(result.state["prefixed"], ["x-a", "x-b"])
        self.assertEqual(result.state["final"], ["x-a-z", "x-b-z"])
        self.assertEqual([step.status for step in result.steps], [StepStatus.SUCCEEDED, StepStatus.SUCCEEDED])
        self.assertEqual(result.metrics["planned_steps"], 2)
        self.assertEqual(result.metrics["succeeded_steps"], 2)

        prefix_record = result.get_step_record("prefix-step")
        suffix_record = result.get_step_record("suffix-step")

        self.assertEqual(prefix_record.inputs, ["a", "b"])
        self.assertEqual(prefix_record.outputs, ["x-a", "x-b"])
        self.assertEqual(prefix_record.config, {"prefix": "x-"})
        self.assertEqual(prefix_record.metrics, {"input_count": 2, "output_count": 2})
        self.assertEqual(prefix_record.output_refs, ["prefixed"])
        self.assertIsNotNone(prefix_record.duration_seconds)
        self.assertGreaterEqual(prefix_record.duration_seconds, 0.0)
        self.assertEqual(
            prefix_record.metadata["observed_resources"],
            {"cluster": "local"},
        )

        self.assertEqual(suffix_record.inputs, ["x-a", "x-b"])
        self.assertEqual(suffix_record.outputs, ["x-a-z", "x-b-z"])
        self.assertEqual(run_store.saved_plans[0]["step_ids"], ["prefix-step", "suffix-step"])
        self.assertEqual(len(run_store.saved_steps), 2)
        self.assertEqual(run_store.saved_results[0]["result"].status, StepStatus.SUCCEEDED)

    def test_executor_records_failure_without_losing_partial_run_state(self) -> None:
        run_store = InMemoryRunStore()
        executor = RuntimeExecutor(
            {
                "utility.prefix": PrefixOperator(),
                "utility.fail": FailingOperator(),
            },
            run_store=run_store,
        )

        result = executor.execute(
            [
                PipelineStep(
                    step_id="prefix-step",
                    operator_id="utility.prefix",
                    inputs=["seed"],
                    outputs=["prefixed"],
                    config={"prefix": "x-"},
                ),
                PipelineStep(
                    step_id="fail-step",
                    operator_id="utility.fail",
                    inputs=["prefixed"],
                    outputs=["never-written"],
                ),
                PipelineStep(
                    step_id="after-fail",
                    operator_id="utility.prefix",
                    inputs=["never-written"],
                    outputs=["unused"],
                    config={"prefix": "ignored-"},
                ),
            ],
            initial_inputs={"seed": ["a"]},
            run_id="run-fail",
        )

        self.assertEqual(result.status, StepStatus.FAILED)
        self.assertEqual(result.state["prefixed"], ["x-a"])
        self.assertNotIn("never-written", result.state)
        self.assertEqual(
            [step.status for step in result.steps],
            [StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.PENDING],
        )
        self.assertEqual(result.error["step_id"], "fail-step")
        self.assertEqual(result.error["operator_id"], "utility.fail")

        failure_record = result.get_step_record("fail-step")
        self.assertEqual(failure_record.status, StepStatus.FAILED)
        self.assertEqual(failure_record.inputs, ["x-a"])
        self.assertEqual(failure_record.error["type"], "RuntimeError")
        self.assertIn("cannot process fail-step", failure_record.error["message"])
        self.assertEqual(run_store.saved_results[0]["result"].status, StepStatus.FAILED)

    def test_executor_can_raise_pipeline_error_after_persisting_result(self) -> None:
        run_store = InMemoryRunStore()
        executor = RuntimeExecutor({"utility.fail": FailingOperator()}, run_store=run_store)

        with self.assertRaises(PipelineExecutionError) as exc_info:
            executor.execute(
                [PipelineStep(step_id="fail-step", operator_id="utility.fail")],
                run_id="run-raise",
                raise_on_error=True,
            )

        self.assertEqual(exc_info.exception.run_id, "run-raise")
        self.assertEqual(exc_info.exception.step_id, "fail-step")
        self.assertEqual(run_store.saved_results[0]["result"].status, StepStatus.FAILED)
