"""从算子类生成轻量级算子目录。

除了内置的 ``dataos_operator.operators`` 包,还支持从**本地插件目录**加载算子:
把环境变量 ``DATAOS_OPERATOR_PLUGINS`` 指向一个或多个目录(以 ``os.pathsep`` 分隔),
目录下每个非下划线开头的 ``.py`` 文件都会被导入,其中的 ``DataOperator`` 具体子类
会被登记进目录 —— 无需改动或重装本包即可新增算子。写本地算子的完整指南见
``dataos_operator.guide()``。
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import pkgutil
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from dataos_operator.core.contracts import DataOperator, OperatorSpec

#: 环境变量名:以 ``os.pathsep`` 分隔的本地算子(插件)目录列表。
PLUGIN_ENV_VAR = "DATAOS_OPERATOR_PLUGINS"


def list_operator_specs(
    package: str | ModuleType = "dataos_operator.operators",
    *,
    plugin_dirs: Iterable[str] | None = None,
    include_plugins: bool = True,
) -> list[OperatorSpec]:
    """扫描内置算子包(以及可选的本地插件目录)并生成算子说明。

    本地插件目录来自 ``plugin_dirs`` 参数;若为 ``None`` 则回退到环境变量
    ``DATAOS_OPERATOR_PLUGINS``(以 ``os.pathsep`` 分隔)。插件里与内置算子**同名**的
    会被忽略(避免意外覆盖内置算子);坏插件(导入报错)会被跳过而不影响其余算子。
    """

    specs: list[OperatorSpec] = []
    for module_name in _iter_module_names(package):
        module = importlib.import_module(module_name)
        for operator_cls in _find_concrete_operator_classes(module):
            specs.append(_build_operator_spec(operator_cls))
    if include_plugins:
        builtin_names = {spec.name for spec in specs}
        for operator_cls in _iter_plugin_operators(plugin_dirs):
            spec = _build_operator_spec(operator_cls)
            if spec.name not in builtin_names:
                specs.append(spec)
                builtin_names.add(spec.name)
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


# --- 本地插件(自定义算子)加载 -------------------------------------------------


def resolve_plugin_dirs(plugin_dirs: Iterable[str] | None = None) -> list[Path]:
    """把 ``plugin_dirs`` 或 ``DATAOS_OPERATOR_PLUGINS`` 解析成存在的目录列表。"""

    if plugin_dirs is not None:
        candidates: Iterable[str] = plugin_dirs
    else:
        candidates = os.environ.get(PLUGIN_ENV_VAR, "").split(os.pathsep)
    dirs: list[Path] = []
    for candidate in candidates:
        text = candidate.strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.is_dir() and path not in dirs:
            dirs.append(path)
    return dirs


def _iter_plugin_operators(
    plugin_dirs: Iterable[str] | None,
) -> list[type[DataOperator[Any, Any]]]:
    operator_classes: list[type[DataOperator[Any, Any]]] = []
    for directory in resolve_plugin_dirs(plugin_dirs):
        dir_str = str(directory)
        if dir_str not in sys.path:
            # 让插件文件能 import 同目录下的下划线辅助模块。
            sys.path.insert(0, dir_str)
        for py_file in sorted(directory.glob("*.py")):
            if py_file.stem.startswith("_"):
                continue
            module = _import_plugin_module(py_file)
            if module is not None:
                operator_classes.extend(_find_concrete_operator_classes(module))
    return operator_classes


def _import_plugin_module(py_file: Path) -> ModuleType | None:
    module_name = f"dataos_operator_plugins_{py_file.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception:
        # 坏插件不应拖垮整个目录扫描;跳过它,其余算子照常可用。
        sys.modules.pop(module_name, None)
        return None
