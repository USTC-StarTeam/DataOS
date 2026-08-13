"""Name -> implementation registry for training backends.

Built-ins are imported lazily so the package imports without heavy backend deps; external
packages can add more via ``[project.entry-points."dataos_train.backends"]``. Launchers have
their own registry in :mod:`dataos_launch.registry` — WHERE a job runs is not this package's
business. Online operators live in :mod:`dataos_train.online`.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata

from .protocols import Backend

_BACKENDS: dict[str, Callable[[], Backend]] = {}

_BUILTINS_LOADED = False
_ENTRYPOINTS_LOADED = False


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    _BACKENDS[name] = factory


def _load_builtins() -> None:
    from .backends.dummy import DummyBackend

    _BACKENDS.setdefault("dummy", DummyBackend)
    # Real backends import torch/llamafactory/verl lazily inside build/collect, so the
    # class itself is cheap to register. A missing optional dep must not break the registry.
    try:
        from .backends.llamafactory import LlamaFactoryBackend

        _BACKENDS.setdefault("llamafactory", LlamaFactoryBackend)
    except Exception:  # noqa: BLE001 - optional
        pass
    try:
        from .backends.verl import VerlBackend

        _BACKENDS.setdefault("verl", VerlBackend)
    except Exception:  # noqa: BLE001 - optional
        pass


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    try:
        eps = metadata.entry_points(group="dataos_train.backends")
    except Exception:  # noqa: BLE001
        return
    for ep in eps:
        if ep.name not in _BACKENDS:
            try:
                _BACKENDS[ep.name] = ep.load()
            except Exception:  # noqa: BLE001
                continue


def _ensure_loaded() -> None:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        _load_builtins()
        _BUILTINS_LOADED = True
    _load_entrypoints()


def get_backend(name: str) -> Backend:
    _ensure_loaded()
    try:
        return _BACKENDS[name]()
    except KeyError:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(_BACKENDS)}") from None


def backend_names() -> list[str]:
    _ensure_loaded()
    return sorted(_BACKENDS)
