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
* ``dataos.project(...)`` — open (or reopen) a research :class:`Project`, the resumable
  top container. Studies live under it: ``project.study(id)``.
* ``Project`` / ``Study`` / ``Experiment`` — containers; ``DataStage`` / ``TrainStage`` /
  ``EvalStage`` — the stages; ``Data`` / ``Model`` / ``Eval`` — the artifacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from dataos._artifact import Artifact, Data, Eval, Model
    from dataos._catalog import catalog
    from dataos._experiment import Experiment
    from dataos._project import Project, project
    from dataos._stage import DataStage, EvalStage, Stage, TrainStage
    from dataos._study import Study
    from dataos.loading import load

__all__ = [
    "Artifact",
    "Data",
    "DataStage",
    "Eval",
    "EvalStage",
    "Experiment",
    "Model",
    "Project",
    "Stage",
    "Study",
    "TrainStage",
    "catalog",
    "load",
    "project",
]

__version__ = "0.2.2"

_LAZY_EXPORTS: dict[str, str] = {
    "catalog": "dataos._catalog",
    "load": "dataos.loading",
    "project": "dataos._project",
    "Project": "dataos._project",
    "Study": "dataos._study",
    "Experiment": "dataos._experiment",
    "Stage": "dataos._stage",
    "DataStage": "dataos._stage",
    "TrainStage": "dataos._stage",
    "EvalStage": "dataos._stage",
    "Artifact": "dataos._artifact",
    "Data": "dataos._artifact",
    "Model": "dataos._artifact",
    "Eval": "dataos._artifact",
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
