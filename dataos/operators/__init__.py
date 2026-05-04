"""Built-in DataOS data operators."""

from dataos.core.operator import Operator
from dataos.core.registry import OperatorRegistry
from dataos.operators.dedup import DedupOperator
from dataos.operators.format_convert import FormatConvertOperator
from dataos.operators.import_dataset import ImportDatasetOperator
from dataos.operators.sample import SampleOperator

__all__ = [
    "DedupOperator",
    "FormatConvertOperator",
    "ImportDatasetOperator",
    "SampleOperator",
    "builtin_operators",
    "build_default_registry",
    "build_registry",
]


def builtin_operators() -> list[Operator]:
    """Return fresh instances of every operator shipped with DataOS."""
    return [
        DedupOperator(),
        FormatConvertOperator(),
        ImportDatasetOperator(),
        SampleOperator(),
    ]


def build_registry(operators: list[Operator]) -> OperatorRegistry:
    """Compose an OperatorRegistry from an explicit operator list."""
    registry = OperatorRegistry()
    for operator in operators:
        registry.register(operator)
    return registry


def build_default_registry() -> OperatorRegistry:
    """Composition of all built-in operators."""
    return build_registry(builtin_operators())
