"""Operator discovery for the planning stage.

``dataos.catalog()`` wraps :func:`dataos_operator.core.list_operator_specs` and
augments each :class:`~dataos_operator.core.OperatorSpec` with the operator's
constructor parameter names (resolved via ``inspect.signature`` on the operator
class ``__init__``). This gives the coding agent the full surface it needs to
plan and parameterize a data pipeline without importing every operator by hand.

Field-key parameters (``id_key``, ``text_key``, ``query_key``, ...) are
normalized so a planner can see the conventional knobs even when an operator
takes them through ``**kwargs``.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any


def catalog() -> list[dict[str, Any]]:
    """Return a list of operator descriptors, each with a ``params`` key.

    Each descriptor::

        {
            "name": str,
            "kind": str,
            "description": str,
            "operator_class": str | None,
            "params": [
                {"name": str, "default": Any, "kind": str, "required": bool},
                ...
            ],
            "accepts_kwargs": bool,
        }
    """
    from dataos_operator.core import list_operator_specs
    from dataos_operator.core.contracts import DEFAULT_FIELD_KEYS

    descriptors: list[dict[str, Any]] = []
    for spec in list_operator_specs():
        descriptors.append(_describe_spec(spec, DEFAULT_FIELD_KEYS))
    return descriptors


def _describe_spec(spec: Any, default_field_keys: dict[str, str]) -> dict[str, Any]:
    params, accepts_kwargs = _resolve_params(spec.operator_class, default_field_keys)
    return {
        "name": spec.name,
        "kind": spec.kind,
        "description": spec.description,
        "operator_class": spec.operator_class,
        "params": params,
        "accepts_kwargs": accepts_kwargs,
    }


def _resolve_params(
    operator_class: str | None,
    default_field_keys: dict[str, str],
) -> tuple[list[dict[str, Any]], bool]:
    if not operator_class:
        return [], False
    try:
        klass = _import_class(operator_class)
        signature = inspect.signature(klass.__init__)
    except (ImportError, AttributeError, ValueError, TypeError):
        return [], False

    params: list[dict[str, Any]] = []
    accepts_kwargs = False
    seen: set[str] = set()
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        has_default = parameter.default is not inspect.Parameter.empty
        params.append(
            {
                "name": name,
                "default": parameter.default if has_default else None,
                "kind": parameter.kind.name.lower(),
                "required": not has_default,
            }
        )
        seen.add(name)

    if accepts_kwargs:
        for field_key, default_value in default_field_keys.items():
            if field_key in seen:
                continue
            params.append(
                {
                    "name": field_key,
                    "default": default_value,
                    "kind": "field_key",
                    "required": False,
                }
            )

    return params, accepts_kwargs


def _import_class(dotted_path: str) -> Any:
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
