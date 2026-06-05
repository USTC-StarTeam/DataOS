"""DataOS public package exports.

DataOS is a library of deterministic executors; the coding agent (Claude Code)
is the orchestrator/planner. There is no embedded LLM or planning agent and no
one-shot ``run()`` — the five stages (plan / data / train / eval / review) are
called imperatively.

Public surface (all resolved lazily via PEP 562 ``__getattr__`` so that
``import dataos`` needs only ``datasets`` + ``dataos_operator``; torch / lm-eval
/ llamafactory are imported on demand inside the stages that use them):

* ``dataos.catalog()`` — discover ``dataos_operator`` operators + their params.
* ``dataos.load(...)`` — load a dataset into a ``dataos_operator.Dataset``.
* ``dataos.experiment(...)`` — start a staged :class:`Experiment`.
* ``Experiment`` / ``Model`` / ``Eval`` — the three core types.
* ``TaskTemplate`` — optional task metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from dataos._catalog import catalog
    from dataos._experiment import Eval, Experiment, Model, experiment
    from dataos.core.task import TaskTemplate
    from dataos.loading import load

__all__ = [
    "Eval",
    "Experiment",
    "Model",
    "TaskTemplate",
    "catalog",
    "experiment",
    "load",
]

__version__ = "0.1.0"

_LAZY_EXPORTS: dict[str, str] = {
    "catalog": "dataos._catalog",
    "load": "dataos.loading",
    "experiment": "dataos._experiment",
    "Experiment": "dataos._experiment",
    "Model": "dataos._experiment",
    "Eval": "dataos._experiment",
    "TaskTemplate": "dataos.core.task",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)
