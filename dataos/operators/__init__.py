"""Built-in DataOS data operators."""

from dataos.core.operator import Operator
from dataos.core.registry import OperatorRegistry
from dataos.operators.category_balance import CategoryBalanceOperator
from dataos.operators.category_tag import CategoryTagOperator
from dataos.operators.curriculum_sort import CurriculumSortOperator
from dataos.operators.dedup import DedupOperator
from dataos.operators.format_convert import FormatConvertOperator
from dataos.operators.import_dataset import ImportDatasetOperator
from dataos.operators.instruction_synthesis_template import InstructionSynthesisTemplateOperator
from dataos.operators.lang_filter import LangFilterOperator
from dataos.operators.length_filter import LengthFilterOperator
from dataos.operators.less import LessSelectOperator
from dataos.operators.mixture import MixtureOperator
from dataos.operators.pii_redact import PiiRedactOperator
from dataos.operators.quality_filter import QualityFilterOperator
from dataos.operators.sample import SampleOperator
from dataos.operators.schema_guard import SchemaGuardOperator

__all__ = [
    "CategoryBalanceOperator",
    "CategoryTagOperator",
    "CurriculumSortOperator",
    "DedupOperator",
    "FormatConvertOperator",
    "ImportDatasetOperator",
    "InstructionSynthesisTemplateOperator",
    "LangFilterOperator",
    "LengthFilterOperator",
    "LessSelectOperator",
    "MixtureOperator",
    "PiiRedactOperator",
    "QualityFilterOperator",
    "SampleOperator",
    "SchemaGuardOperator",
    "builtin_operators",
    "build_default_registry",
    "build_registry",
]


def builtin_operators() -> list[Operator]:
    """Return fresh instances of every operator shipped with DataOS."""
    return [
        DedupOperator(),
        SampleOperator(),
        SchemaGuardOperator(),
        PiiRedactOperator(),
        LengthFilterOperator(),
        LangFilterOperator(),
        QualityFilterOperator(),
        CategoryTagOperator(),
        CategoryBalanceOperator(),
        CurriculumSortOperator(),
        MixtureOperator(),
        InstructionSynthesisTemplateOperator(),
        FormatConvertOperator(),
        ImportDatasetOperator(),
        LessSelectOperator(),
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
