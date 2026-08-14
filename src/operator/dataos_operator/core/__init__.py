"""核心入口：DataOS 算子共享契约。"""

from __future__ import annotations

import importlib

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
    "OperatorState",
    "PolicyAction",
    "PolicyRuntime",
    "Record",
    "RecordOrDataset",
    "RecordOperator",
    "RefinerOperator",
    "ReweighterOperator",
    "RuntimeContext",
    "RuntimeMode",
    "SelectorOperator",
    "ArtifactRef",
    "BackwardSignal",
    "BatchView",
    "ForwardSignal",
    "get_field",
    "is_dataset",
    "is_multi_dataset",
    "is_record",
    "list_operator_specs",
    "require_field",
    "set_field",
    "validate_operator_layout",
]

_LAZY_EXPORTS = {
    "ArtifactRef": ("dataos_operator.runtime", "ArtifactRef"),
    "BackwardSignal": ("dataos_operator.runtime", "BackwardSignal"),
    "BatchView": ("dataos_operator.runtime", "BatchView"),
    "ForwardSignal": ("dataos_operator.runtime", "ForwardSignal"),
    "OperatorState": ("dataos_operator.runtime", "OperatorState"),
    "PolicyAction": ("dataos_operator.runtime", "PolicyAction"),
    "PolicyRuntime": ("dataos_operator.runtime", "PolicyRuntime"),
    "RuntimeContext": ("dataos_operator.runtime", "RuntimeContext"),
    "RuntimeMode": ("dataos_operator.runtime", "RuntimeMode"),
}


def __getattr__(name: str) -> object:
    if name == "Dataset":
        from dataos_operator.dataset import Dataset

        return Dataset
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
