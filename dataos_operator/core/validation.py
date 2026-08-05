"""算子布局约定的校验工具。"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

from dataos_operator.core.contracts import OPERATOR_KINDS, DataOperator


@dataclass(frozen=True)
class OperatorLayoutIssue:
    """一条布局校验问题。"""

    message: str
    module: str | None = None
    operator: str | None = None


def validate_operator_layout(
    package: str | ModuleType = "dataos_operator.operators",
) -> list[OperatorLayoutIssue]:
    """校验轻量级算子布局不变量。"""

    issues: list[OperatorLayoutIssue] = []
    for module_name in _iter_operator_module_names(package):
        module_stem = module_name.rsplit(".", 1)[-1]
        if not _is_snake_case(module_stem):
            issues.append(
                OperatorLayoutIssue(
                    message=f"算子文件名必须是 snake_case：{module_stem!r}",
                    module=module_name,
                )
            )

        module = _import_module(module_name, issues)
        if module is None:
            continue

        operator_classes = _find_concrete_operator_classes(module)
        if len(operator_classes) == 0:
            issues.append(
                OperatorLayoutIssue(
                    message="算子模块必须定义一个具体算子类",
                    module=module_name,
                )
            )
            continue
        if len(operator_classes) > 1:
            issues.append(
                OperatorLayoutIssue(
                    message="一个算子模块只能定义一个具体算子类",
                    module=module_name,
                )
            )
            continue

        operator_cls = operator_classes[0]
        expected_kind = _expected_kind_from_module(module_name)
        if expected_kind is None:
            issues.append(
                OperatorLayoutIssue(
                    message="算子模块必须放在 operators/<kind>/ 目录下",
                    module=module_name,
                    operator=operator_cls.__name__,
                )
            )
        elif operator_cls.operator_kind != expected_kind:
            issues.append(
                OperatorLayoutIssue(
                    message=(
                        f"算子 kind {operator_cls.operator_kind!r} 必须等于目录名 {expected_kind!r}"
                    ),
                    module=module_name,
                    operator=operator_cls.__name__,
                )
            )

        if not inspect.getdoc(operator_cls):
            issues.append(
                OperatorLayoutIssue(
                    message="算子类必须有 docstring",
                    module=module_name,
                    operator=operator_cls.__name__,
                )
            )

    return issues


def _iter_operator_module_names(package: str | ModuleType) -> list[str]:
    package_module = _import_package(package)
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


def _import_package(package: str | ModuleType) -> ModuleType:
    if isinstance(package, str):
        return importlib.import_module(package)
    return package


def _import_module(
    module_name: str,
    issues: list[OperatorLayoutIssue],
) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        issues.append(
            OperatorLayoutIssue(
                message=f"算子模块导入失败：{exc}",
                module=module_name,
            )
        )
        return None


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


def _expected_kind_from_module(module_name: str) -> str | None:
    parts = module_name.split(".")
    try:
        operators_index = parts.index("operators")
    except ValueError:
        return None
    if len(parts) <= operators_index + 1:
        return None
    kind = parts[operators_index + 1]
    if kind not in OPERATOR_KINDS:
        return None
    return kind


def _is_snake_case(value: str) -> bool:
    return re.fullmatch(r"[a-z][a-z0-9_]*", value) is not None
