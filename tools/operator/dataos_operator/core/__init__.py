"""核心入口：DataOS 算子共享契约。"""

from __future__ import annotations

from dataos_operator.core.catalog import list_operator_specs
from dataos_operator.core.contracts import (
    OPERATOR_KINDS,
    ClustererOperator,
    DataOperator,
    DatasetInput,
    DatasetOperator,
    EvaluatorOperator,
    GeneratorOperator,
    MapperOperator,
    MixerOperator,
    OperatorKind,
    OperatorSpec,
    Record,
    RecordOperator,
    RecordOrDataset,
    RefinerOperator,
    ReweighterOperator,
    SelectorOperator,
    get_field,
    is_dataset,
    is_multi_dataset,
    is_record,
    require_field,
    set_field,
)
from dataos_operator.core.validation import OperatorLayoutIssue, validate_operator_layout

__all__ = [
    "Dataset",
    "DatasetOperator",
    "DataOperator",
    "DatasetInput",
    "ClustererOperator",
    "EvaluatorOperator",
    "GeneratorOperator",
    "MapperOperator",
    "MixerOperator",
    "OPERATOR_KINDS",
    "OperatorKind",
    "OperatorLayoutIssue",
    "OperatorSpec",
    "Record",
    "RecordOrDataset",
    "RecordOperator",
    "RefinerOperator",
    "ReweighterOperator",
    "SelectorOperator",
    "get_field",
    "is_dataset",
    "is_multi_dataset",
    "is_record",
    "list_operator_specs",
    "require_field",
    "set_field",
    "validate_operator_layout",
]


def __getattr__(name: str) -> object:
    if name == "Dataset":
        from dataos_operator.dataset import Dataset

        return Dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
