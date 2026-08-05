"""Name -> implementation registries for backends and launchers.

Built-ins are imported lazily so the package imports without heavy backend deps; external
packages can add more via ``[project.entry-points."dataos_train.backends"]`` etc. Online
operators live in their own registry: :mod:`dataos_train.online`.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any

from .protocols import Backend, Launcher

_BACKENDS: dict[str, Callable[[], Backend]] = {}
_LAUNCHERS: dict[str, Callable[..., Launcher]] = {}

_BUILTINS_LOADED = False
_ENTRYPOINTS_LOADED = False


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    _BACKENDS[name] = factory


def register_launcher(name: str, factory: Callable[..., Launcher]) -> None:
    _LAUNCHERS[name] = factory


def _load_builtins() -> None:
    from .backends.dummy import DummyBackend
    from .launchers.local import LocalLauncher
    from .launchers.slurm import SlurmLauncher

    _LAUNCHERS.setdefault("local", LocalLauncher)
    _LAUNCHERS.setdefault("slurm", SlurmLauncher)
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
    for group, target in (
        ("dataos_train.backends", _BACKENDS),
        ("dataos_train.launchers", _LAUNCHERS),
    ):
        try:
            eps = metadata.entry_points(group=group)
        except Exception:  # noqa: BLE001
            continue
        for ep in eps:
            if ep.name not in target:
                try:
                    target[ep.name] = ep.load()
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


def get_launcher(name: str, **cfg: Any) -> Launcher:
    _ensure_loaded()
    try:
        return _LAUNCHERS[name](**cfg)
    except KeyError:
        raise KeyError(f"unknown launcher {name!r}; known: {sorted(_LAUNCHERS)}") from None


def backend_names() -> list[str]:
    _ensure_loaded()
    return sorted(_BACKENDS)
