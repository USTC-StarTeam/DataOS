from __future__ import annotations

import unittest

from dataos.core import Operator, OperatorContext, OperatorResult, OperatorSpec


class EchoOperator:
    spec = OperatorSpec(
        operator_id="utility.echo",
        name="Echo",
        category="utility",
        input_type="records",
        output_type="records",
        config_schema={"type": "object", "properties": {"prefix": {"type": "string"}}},
        deterministic=True,
    )

    def execute(self, context: OperatorContext) -> OperatorResult:
        prefix = context.config.get("prefix", "")
        outputs = [f"{prefix}{item}" for item in context.inputs]
        return OperatorResult(
            outputs=outputs,
            metrics={"input_count": len(context.inputs), "output_count": len(outputs)},
            metadata={"run_id": context.run_id, "step_id": context.step_id},
        )


class DataOSOperatorTests(unittest.TestCase):
    def test_operator_contract_supports_registry_and_runtime_usage(self) -> None:
        operator = EchoOperator()
        registry = {operator.spec.operator_id: operator}

        selected = registry["utility.echo"]
        self.assertIsInstance(selected, Operator)

        result = selected.execute(
            OperatorContext(
                run_id="run-001",
                step_id="step-echo",
                inputs=["a", "b"],
                config={"prefix": "x-"},
            )
        )

        self.assertEqual(result.outputs, ["x-a", "x-b"])
        self.assertEqual(result.metrics, {"input_count": 2, "output_count": 2})
        self.assertEqual(result.metadata, {"run_id": "run-001", "step_id": "step-echo"})

    def test_operator_spec_rejects_blank_required_fields(self) -> None:
        with self.assertRaises(ValueError):
            OperatorSpec(
                operator_id=" ",
                name="Echo",
                category="utility",
                input_type="records",
                output_type="records",
            )

    def test_context_and_result_copy_mutable_payloads(self) -> None:
        inputs = ["seed"]
        config = {"prefix": "pre-"}
        metrics = {"count": 1}

        context = OperatorContext(run_id="run-1", step_id="step-1", inputs=inputs, config=config)
        result = OperatorResult(outputs=inputs, metrics=metrics)

        inputs.append("mutated")
        config["prefix"] = "changed"
        metrics["count"] = 2

        self.assertEqual(context.inputs, ["seed"])
        self.assertEqual(context.config, {"prefix": "pre-"})
        self.assertEqual(result.outputs, ["seed"])
        self.assertEqual(result.metrics, {"count": 1})
