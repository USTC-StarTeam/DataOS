"""Self-documenting usage catalog for DataOS operators.

``usage()`` is the single entry point a *consumer* of this package calls to learn
how operators are used, without needing the source repository checked out. The
per-operator documentation is derived by introspection from the operator classes
themselves (docstring + ``__init__`` signature + ``operator_kind``), so it can
never drift from the installed code. A curated framework-level guide
(``_usage_guide.md``, shipped inside the package) is prepended to explain the
two ways operators are used (offline / online) and the online-runtime config schema.
"""

from __future__ import annotations

import importlib
import importlib.resources
import inspect
import re
from typing import Any, Literal, cast

from dataos_operator.core.catalog import list_operator_specs
from dataos_operator.core.contracts import DataOperator, OperatorKind, OperatorSpec

__all__ = ["catalog", "usage", "guide", "reference", "operator_docs", "OperatorDoc"]

# Runtime lifecycle hooks an *online* (policy-driven) operator overrides. A pure
# offline operator leaves all of these at the ``DataOperator`` base defaults.
_RUNTIME_HOOKS = (
    "prepare",
    "before_batch",
    "after_forward",
    "before_backward",
    "after_backward",
    "on_update_boundary",
    "finalize",
)

# The ``run(data)`` input/output contract per operator kind.
_RUN_IO: dict[OperatorKind, tuple[str, str]] = {
    "selector": ("Dataset | Sequence[Dataset]", "Dataset"),
    "mixer": ("Sequence[Dataset]", "Dataset"),
    "mapper": ("Record | Dataset", "Dataset"),
    "evaluator": ("Record | Dataset", "Dataset"),
    "refiner": ("Record | Dataset", "Dataset"),
    "reweighter": ("Record | Dataset", "Dataset"),
    "clusterer": ("Dataset | Sequence[Dataset]", "Dataset"),
    "generator": ("Dataset | Sequence[Dataset]", "Dataset"),
}

_GUIDE_RESOURCE = "_usage_guide.md"

Mode = Literal["offline", "online"]
Param = dict[str, Any]
OperatorDoc = dict[str, Any]


def catalog(
    *,
    kind: OperatorKind | None = None,
    mode: Mode | None = None,
    fmt: Literal["markdown", "dict"] = "markdown",
) -> str | list[dict[str, Any]]:
    """Tier-1 index: one compact line per operator — name, kind, mode, effect.

    Cheap enough to drop into a planning/main-agent prompt: it answers *which
    operators exist and what each does*, not how to call them. Drill into
    :func:`usage` for a single operator's parameters and config.

    Args:
        kind: Filter by operator kind (``"selector"``, ``"mixer"``, ...).
        mode: Filter by execution mode (``"offline"`` or ``"online"``).
        fmt: ``"markdown"`` for a bullet list, ``"dict"`` for ``{name, kind, mode, effect}`` rows.
    """
    rows = [
        {
            "name": entry["name"],
            "kind": entry["kind"],
            "mode": entry["mode"],
            "effect": entry["summary"],
        }
        for entry in operator_docs(kind=kind, mode=mode)
    ]
    if fmt == "dict":
        return rows
    lines = [
        f"- `{row['name']}` ({row['kind']} · {row['mode']}) — {row['effect']}".rstrip(" —")
        for row in sorted(rows, key=lambda r: (r["mode"], r["kind"], r["name"]))
    ]
    header = "# Operator catalog (call `usage('<name>')` for how to use one)\n"
    return header + "\n".join(lines) + "\n"


def usage(
    name: str,
    *,
    fmt: Literal["markdown", "dict"] = "markdown",
) -> str | dict[str, Any]:
    """Tier-2 detail: how to use *one* operator (params, signature, config example).

    This is the view a sub-agent loads for the single operator it was assigned —
    not the whole manual. Pair it with :func:`guide` for the framework how-to.

    Args:
        name: Operator registry name (e.g. ``"loss_selector"``) or class name
            (``"LossSelector"``, case-insensitive).
        fmt: ``"markdown"`` for a ready-to-prompt section, ``"dict"`` for the
            structured entry.

    Raises:
        ValueError: If no operator matches ``name`` (the message points to ``catalog()``).
    """
    matches = operator_docs(name=name)
    if not matches:
        raise ValueError(
            f"Unknown operator {name!r}. Call dataos_operator.catalog() to list operators."
        )
    entry = matches[0]
    if fmt == "dict":
        return entry
    return _render_operator(entry) + "\n"


def guide() -> str:
    """Tier-0: the curated framework guide (data model, offline vs online use,
    and the online-runtime config schema). Loaded once by a sub-agent; shipped as
    package data so it is available from the installed wheel alone."""
    return _load_guide()


def reference(*, kind: OperatorKind | None = None, mode: Mode | None = None) -> str:
    """Full manual: guide + every operator's detail. Intended for generating an
    offline docs file — do NOT feed this into an agent context (it is large; use
    :func:`catalog` + :func:`usage` for progressive disclosure)."""
    docs = operator_docs(kind=kind, mode=mode)
    return _render_markdown(docs, guide=_load_guide())


def operator_docs(
    *,
    name: str | None = None,
    kind: OperatorKind | None = None,
    mode: Mode | None = None,
) -> list[OperatorDoc]:
    """Return the introspected per-operator documentation entries (structured)."""
    entries: list[OperatorDoc] = []
    for spec in list_operator_specs():
        entry = _build_entry(spec)
        if entry is None:
            continue
        if kind is not None and entry["kind"] != kind:
            continue
        if mode is not None and entry["mode"] != mode:
            continue
        if name is not None and not _name_matches(entry, name):
            continue
        entries.append(entry)
    return entries


# --------------------------------------------------------------------------- #
# Introspection helpers
# --------------------------------------------------------------------------- #


def _build_entry(spec: OperatorSpec) -> OperatorDoc | None:
    operator_cls = _resolve_class(spec.operator_class)
    if operator_cls is None:
        return None
    summary, implementation, dependency = _parse_docstring(spec.description)
    online = _is_online(operator_cls)
    run_input, run_output = _RUN_IO.get(spec.kind, ("Dataset", "Dataset"))
    params = _init_params(operator_cls)
    entry: OperatorDoc = {
        "name": spec.name,
        "class_name": operator_cls.__name__,
        "kind": spec.kind,
        "mode": "online" if online else "offline",
        "dependency": dependency,
        "summary": summary,
        "implementation": implementation,
        "params": params,
        "run_input": run_input,
        "run_output": run_output,
        "supports_offline_run": True,
        "operator_class": spec.operator_class,
    }
    if online:
        entry["online_config_example"] = {
            "operator": spec.name,
            "kwargs": {p["name"]: _placeholder(p) for p in params if p["required"]},
        }
    return entry


def _resolve_class(dotted_path: str | None) -> type[DataOperator[Any, Any]] | None:
    if not dotted_path:
        return None
    module_name, _, qualname = dotted_path.rpartition(".")
    if not module_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    obj = getattr(module, qualname, None)
    if isinstance(obj, type) and issubclass(obj, DataOperator):
        return obj
    return None


def _is_online(operator_cls: type[DataOperator[Any, Any]]) -> bool:
    """True when the operator overrides any runtime lifecycle hook."""
    for hook in _RUNTIME_HOOKS:
        defining = _defining_class(operator_cls, hook)
        if defining is not None and defining is not DataOperator:
            return True
    return False


def _defining_class(operator_cls: type[Any], method: str) -> type[Any] | None:
    for klass in operator_cls.__mro__:
        if method in vars(klass):
            return klass
    return None


def _init_params(operator_cls: type[DataOperator[Any, Any]]) -> list[Param]:
    try:
        signature = inspect.signature(operator_cls)
    except (TypeError, ValueError):
        return []
    params: list[Param] = []
    for parameter in signature.parameters.values():
        if parameter.name == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        required = parameter.default is inspect.Parameter.empty
        param: Param = {
            "name": parameter.name,
            "type": _annotation_str(parameter.annotation),
            "required": required,
        }
        if not required:
            param["default"] = parameter.default
        params.append(param)
    return params


def _annotation_str(annotation: Any) -> str | None:
    if annotation is inspect.Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    return inspect.formatannotation(annotation)


_SECTION_RE = re.compile(r"^(Implementation|Dependency)\s*:\s*", re.IGNORECASE)


def _parse_docstring(docstring: str) -> tuple[str, str, str]:
    """Split a structured operator docstring into (summary, implementation, dependency)."""
    sections: dict[str, list[str]] = {"summary": [], "implementation": [], "dependency": []}
    current = "summary"
    for raw_line in (docstring or "").splitlines():
        match = _SECTION_RE.match(raw_line.strip())
        if match:
            current = match.group(1).lower()
            remainder = raw_line.strip()[match.end() :].strip()
            if remainder:
                sections[current].append(remainder)
            continue
        sections[current].append(raw_line.strip())
    return (
        _join(sections["summary"]),
        _join(sections["implementation"]),
        _join(sections["dependency"]),
    )


def _join(lines: list[str]) -> str:
    return " ".join(line for line in lines if line).strip()


def _placeholder(param: Param) -> Any:
    type_name = (param.get("type") or "").lower()
    if "int" in type_name:
        return 0
    if "float" in type_name:
        return 0.0
    if "bool" in type_name:
        return False
    return "..."


def _name_matches(entry: OperatorDoc, name: str) -> bool:
    needle = name.strip().lower()
    return needle in {entry["name"].lower(), entry["class_name"].lower()}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_markdown(docs: list[OperatorDoc], *, guide: str | None) -> str:
    blocks: list[str] = []
    if guide:
        blocks.append(guide.strip())
    if docs:
        blocks.append("# Operators\n")
        blocks.extend(_render_operator(entry) for entry in docs)
    return "\n\n".join(blocks).strip() + "\n"


def _render_operator(entry: OperatorDoc) -> str:
    lines: list[str] = []
    lines.append(f"## `{entry['name']}` — {entry['kind']} · {entry['mode']}")
    if entry["summary"]:
        lines.append(entry["summary"])
    if entry["dependency"]:
        lines.append(f"- **Dependency:** {entry['dependency']}")
    if entry["implementation"]:
        lines.append(f"- **Implementation:** {entry['implementation']}")
    lines.append(f"- **run():** `{entry['run_input']}` → `{entry['run_output']}`")

    signature = _render_signature(entry)
    lines.append(f"- **Constructor:** `{signature}`")
    if entry["params"]:
        lines.append("")
        lines.append("| param | type | required | default |")
        lines.append("| --- | --- | --- | --- |")
        for param in entry["params"]:
            default = "—" if param["required"] else f"`{param.get('default')!r}`"
            lines.append(
                f"| `{param['name']}` | {param.get('type') or '—'} "
                f"| {'yes' if param['required'] else 'no'} | {default} |"
            )

    if entry["mode"] == "online":
        example = entry.get("online_config_example")
        lines.append("")
        lines.append("Online (training-loop) use — declare it in the DataOS runtime config:")
        lines.append("```json")
        lines.append(_compact_json(example))
        lines.append("```")
    else:
        lines.append("")
        lines.append(f"```python\n{entry['class_name']}(...).run(dataset)\n```")
    return "\n".join(lines)


def _render_signature(entry: OperatorDoc) -> str:
    parts: list[str] = []
    for param in entry["params"]:
        text = param["name"]
        if param.get("type"):
            text += f": {param['type']}"
        if not param["required"]:
            text += f" = {param.get('default')!r}"
        parts.append(text)
    return f"{entry['class_name']}({', '.join(parts)})"


def _compact_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# Package resources
# --------------------------------------------------------------------------- #


def _load_guide() -> str:
    try:
        resource = importlib.resources.files("dataos_operator").joinpath(_GUIDE_RESOURCE)
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return ""


def _package_version() -> str:
    module = importlib.import_module("dataos_operator")
    return cast("str", getattr(module, "__version__", "unknown"))
