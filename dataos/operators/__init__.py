"""Built-in DataOS data operators grouped by operator category."""

from dataos.core.operator import Operator
from dataos.core.registry import OperatorRegistry
from dataos.operators.evaluator import EvaluateLLMPairOperator, EvaluateTextQualityOperator
from dataos.operators.filter import FilterByTextLengthOperator, FilterExactDedupOperator
from dataos.operators.generator import GenerateRewrittenAnswerOperator, GenerateRewrittenQueryOperator
from dataos.operators.mapper import MapToAlpacaOperator, MapToShareGPTOperator
from dataos.operators.mix import MergeDatasetsOperator
from dataos.operators.selector import SelectRandomSubsetOperator, SelectTopByScoreOperator

__all__ = [
    "EvaluateLLMPairOperator",
    "EvaluateTextQualityOperator",
    "FilterByTextLengthOperator",
    "FilterExactDedupOperator",
    "GenerateRewrittenAnswerOperator",
    "GenerateRewrittenQueryOperator",
    "MapToAlpacaOperator",
    "MapToShareGPTOperator",
    "MergeDatasetsOperator",
    "SelectRandomSubsetOperator",
    "SelectTopByScoreOperator",
    "builtin_operators",
    "build_default_registry",
    "build_registry",
]


def builtin_operators() -> list[Operator]:
    return [
        MapToAlpacaOperator(),
        MapToShareGPTOperator(),
        FilterExactDedupOperator(),
        FilterByTextLengthOperator(),
        EvaluateTextQualityOperator(),
        EvaluateLLMPairOperator(),
        SelectRandomSubsetOperator(),
        SelectTopByScoreOperator(),
        GenerateRewrittenQueryOperator(),
        GenerateRewrittenAnswerOperator(),
    ]


def build_registry(operators: list[Operator]) -> OperatorRegistry:
    registry = OperatorRegistry()
    for operator in operators:
        registry.register(operator)
    return registry


def build_default_registry() -> OperatorRegistry:
    return build_registry(builtin_operators())
