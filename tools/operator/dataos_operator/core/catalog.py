"""从算子类生成轻量级算子目录。"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterable
from types import ModuleType
from typing import Any, cast

from dataos_operator.core.contracts import DataOperator, OperatorSpec


def list_operator_specs(
    package: str | ModuleType = "dataos_operator.operators",
) -> list[OperatorSpec]:
    """扫描算子包并生成算子说明。"""

    specs: list[OperatorSpec] = []
    for module_name in _iter_module_names(package):
        module = importlib.import_module(module_name)
        for operator_cls in _find_concrete_operator_classes(module):
            specs.append(_build_operator_spec(operator_cls))
    return sorted(specs, key=lambda spec: (spec.kind, spec.name))


def _iter_module_names(package: str | ModuleType) -> list[str]:
    package_module = importlib.import_module(package) if isinstance(package, str) else package
    package_paths = cast(Iterable[str] | None, getattr(package_module, "__path__", None))
    if package_paths is None:
        return []
    return sorted(
        module_info.name
        for module_info in pkgutil.walk_packages(
            list(package_paths),
            prefix=package_module.__name__ + ".",
        )
        if not module_info.ispkg
    )


def _find_concrete_operator_classes(module: ModuleType) -> list[type[DataOperator[Any, Any]]]:
    operator_classes: list[type[DataOperator[Any, Any]]] = []
    for _, member in inspect.getmembers(module, inspect.isclass):
        if member.__module__ != module.__name__:
            continue
        if not issubclass(member, DataOperator):
            continue
        if inspect.isabstract(member):
            continue
        operator_classes.append(member)
    return operator_classes


def _build_operator_spec(operator_cls: type[DataOperator[Any, Any]]) -> OperatorSpec:
    module_stem = operator_cls.__module__.rsplit(".", 1)[-1]
    return OperatorSpec(
        name=operator_cls.operator_name or module_stem,
        kind=operator_cls.operator_kind,
        description=inspect.getdoc(operator_cls) or "",
        operator_class=f"{operator_cls.__module__}.{operator_cls.__qualname__}",
    )
