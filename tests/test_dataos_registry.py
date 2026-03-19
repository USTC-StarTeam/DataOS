from __future__ import annotations

import unittest

from dataos.core import OperatorContext, OperatorResult, OperatorSpec
from dataos.eval import SimpleSFTEvaluator
from dataos.registry import (
    DuplicateEvaluatorError,
    DuplicateOperatorError,
    EvaluatorNotFoundError,
    EvaluatorRegistry,
    OperatorNotFoundError,
    OperatorRegistry,
)
from dataos.runtime import OperatorRegistryLike


class EchoOperator:
    spec = OperatorSpec(
        operator_id="utility.echo",
        name="Echo",
        category="utility",
        input_type="records",
        output_type="records",
    )

    def execute(self, context: OperatorContext) -> OperatorResult:
        return OperatorResult(outputs=list(context.inputs))


class CountOperator:
    spec = OperatorSpec(
        operator_id="utility.count",
        name="Count",
        category="utility",
        input_type="records",
        output_type="metrics",
    )

    def execute(self, context: OperatorContext) -> OperatorResult:
        return OperatorResult(metrics={"count": len(context.inputs)})


class OperatorRegistryTests(unittest.TestCase):
    def test_registry_get_returns_registered_operator_instance(self) -> None:
        echo = EchoOperator()
        registry = OperatorRegistry([echo])

        selected = registry.get("utility.echo")

        self.assertIs(selected, echo)
        self.assertIs(registry.get_operator("utility.echo"), echo)
        self.assertTrue(registry.supports("utility.echo"))
        self.assertIn("utility.echo", registry)
        self.assertEqual(len(registry), 1)
        self.assertIsInstance(registry, OperatorRegistryLike)

    def test_registry_get_spec_returns_registered_operator_metadata(self) -> None:
        echo = EchoOperator()
        registry = OperatorRegistry([echo])

        spec = registry.get_spec("utility.echo")

        self.assertIs(spec, echo.spec)
        self.assertEqual(spec.operator_id, "utility.echo")

    def test_registry_discover_filters_allowed_operators_for_planner(self) -> None:
        registry = OperatorRegistry([EchoOperator(), CountOperator()])

        specs = registry.discover(
            allowed_operator_ids=[
                "utility.count",
                "utility.echo",
                "utility.count",
            ]
        )

        self.assertEqual(
            [spec.operator_id for spec in specs],
            ["utility.count", "utility.echo"],
        )
        self.assertEqual(
            registry.list_operator_ids(["utility.count", "utility.echo"]),
            ["utility.count", "utility.echo"],
        )

    def test_registry_rejects_duplicate_operator_ids(self) -> None:
        registry = OperatorRegistry([EchoOperator()])

        with self.assertRaises(DuplicateOperatorError):
            registry.register(EchoOperator())

    def test_registry_raises_for_unknown_allowed_operator(self) -> None:
        registry = OperatorRegistry([EchoOperator()])

        with self.assertRaises(OperatorNotFoundError):
            registry.discover(["utility.echo", "utility.missing"])


class EvaluatorRegistryTests(unittest.TestCase):
    def test_registry_get_returns_registered_evaluator_instance(self) -> None:
        evaluator = SimpleSFTEvaluator()
        registry = EvaluatorRegistry([evaluator])

        selected = registry.get("simple_sft_eval")

        self.assertIs(selected, evaluator)
        self.assertTrue(registry.supports("simple_sft_eval"))
        self.assertIn("simple_sft_eval", registry)
        self.assertEqual(
            [spec.evaluator_id for spec in registry.discover(["simple_sft_eval"])],
            ["simple_sft_eval"],
        )

    def test_registry_rejects_duplicate_evaluator_ids(self) -> None:
        registry = EvaluatorRegistry([SimpleSFTEvaluator()])

        with self.assertRaises(DuplicateEvaluatorError):
            registry.register(SimpleSFTEvaluator())

    def test_registry_raises_for_unknown_evaluator(self) -> None:
        registry = EvaluatorRegistry([SimpleSFTEvaluator()])

        with self.assertRaises(EvaluatorNotFoundError):
            registry.discover(["simple_sft_eval", "missing_eval"])
