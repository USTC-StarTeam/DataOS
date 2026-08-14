"""Name -> launcher registry.

Built-ins load lazily so importing this package costs nothing; external packages add more
through the ``dataos_launch.launchers`` entry-point group. The registry is the seam that
lets the SAME rendered job run under Slurm on a cluster or as a subprocess on a workstation
— callers name a launcher, they never import one.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any

from .launchers.base import Launcher

_LAUNCHERS: dict[str, Callable[..., Launcher]] = {}
_BUILTINS_LOADED = False
_ENTRYPOINTS_LOADED = False

ENTRY_POINT_GROUP = "dataos_launch.launchers"


def register_launcher(name: str, factory: Callable[..., Launcher]) -> None:
    _LAUNCHERS[name] = factory


def _load_builtins() -> None:
    from .launchers.local import LocalLauncher
    from .launchers.slurm import SlurmLauncher

    _LAUNCHERS.setdefault("local", LocalLauncher)
    _LAUNCHERS.setdefault("slurm", SlurmLauncher)


def _load_entrypoints() -> None:
    global _ENTRYPOINTS_LOADED
    if _ENTRYPOINTS_LOADED:
        return
    _ENTRYPOINTS_LOADED = True
    try:
        eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001
        return
    for ep in eps:
        if ep.name not in _LAUNCHERS:
            try:
                _LAUNCHERS[ep.name] = ep.load()
            except Exception:  # noqa: BLE001
                continue


def _ensure_loaded() -> None:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        _load_builtins()
        _BUILTINS_LOADED = True
    _load_entrypoints()


def get_launcher(name: str, **cfg: Any) -> Launcher:
    _ensure_loaded()
    try:
        return _LAUNCHERS[name](**cfg)
    except KeyError:
        raise KeyError(f"unknown launcher {name!r}; known: {sorted(_LAUNCHERS)}") from None


def launcher_names() -> list[str]:
    _ensure_loaded()
    return sorted(_LAUNCHERS)
