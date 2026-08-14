"""Registration, resolution, and discovery of online operators.

Torch-free by design: this module is imported by ``validate()`` on the submit side, so it
must never pull the training stack. External ops are discovered through the entry-point
group ``dataos_train.online_ops`` (duck-typed, no import coupling).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

_ENTRY_POINT_GROUP = "dataos_train.online_ops"

_HOOK_FACTORIES: dict[str, Callable[..., Any]] = {}


def register_online_op(name: str, factory: Callable[..., Any]) -> None:
    _HOOK_FACTORIES[name] = factory


def online_op_names() -> list[str]:
    return sorted(_HOOK_FACTORIES)


def known_online_op_names() -> list[str]:
    """Registered factories plus names visible in the entry-point group.

    The entry-point half is metadata-only (loads nothing), so this stays torch-free
    and safe for ``validate()`` on the submit side.
    """
    try:
        from importlib.metadata import entry_points

        external = [ep.name for ep in entry_points(group=_ENTRY_POINT_GROUP)]
    except Exception:  # noqa: BLE001
        external = []
    return sorted(set(online_op_names()) | set(external))


def _load_entry_point(name: str) -> bool:
    """Discover an external online op via the ``dataos_train.online_ops`` entry-point group
    (e.g. ops shipped by dataos_operator) — duck-typed protocol, no import coupling."""
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group=_ENTRY_POINT_GROUP):
            if ep.name == name:
                register_online_op(name, ep.load())
                return True
    except Exception:  # noqa: BLE001
        return False
    return False


def resolve_online_ops(specs: Sequence[dict]) -> list[Any]:
    """Turn a list of ``OnlineOpSpec`` dicts into hook instances.

    Each factory is called as ``factory(hook=spec.get("hook"), **params)``; ``hook=None``
    lets the op pick its own default hook point, and an unsupported explicit hook must
    raise ``ValueError`` at instantiation.
    """
    hooks: list[Any] = []
    for spec in specs:
        name = spec["op"]
        if name not in _HOOK_FACTORIES and not _load_entry_point(name):
            raise KeyError(f"unknown online op {name!r}; known: {known_online_op_names()}")
        params = dict(spec.get("params") or {})
        hooks.append(_HOOK_FACTORIES[name](hook=spec.get("hook"), **params))
    return hooks


def probe_online_op(name: str) -> bool | None:
    """Best-effort resolvability probe for ``validate()`` (torch-free, loads nothing).

    Returns ``True`` if the op is registered in-module or visible in the entry-point
    group, ``False`` if the scan completed and the op is definitely unknown, and ``None``
    if the entry-point metadata scan failed and resolvability cannot be confirmed.
    """
    if name in _HOOK_FACTORIES:
        return True
    try:
        from importlib.metadata import entry_points

        return any(ep.name == name for ep in entry_points(group=_ENTRY_POINT_GROUP))
    except Exception:  # noqa: BLE001
        return None
