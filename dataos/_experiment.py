"""Experiment — a single staged run composed of three Stage objects.

An experiment is a container: it owns a workspace directory and composes a
:class:`~dataos._stage.DataStage` / :class:`~dataos._stage.TrainStage` /
:class:`~dataos._stage.EvalStage` (built from its ``spec``). The coding agent
drives the stages imperatively — ``exp.data.run()`` -> ``exp.train.run()`` ->
``exp.eval.run()`` — there is no ``run()`` that chains them. ``report()`` writes a
flat ``report.json`` recomposed from each stage / artifact ``summary()``.

The artifact types live in :mod:`dataos._artifact`; the stages in
:mod:`dataos._stage`. Heavy deps stay lazy (inside the stages' ``run``), so
``import dataos`` needs only ``datasets`` + ``dataos_operator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataos._ids import _make_id, _utc_now_iso
from dataos._stage import DataStage, EvalStage, TrainStage
from dataos.workspace import Workspace


@dataclass
class Experiment:
    exp_id: str
    workspace: Workspace
    name: str = ""
    spec: dict[str, Any] | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    data: DataStage = field(init=False)
    train: TrainStage = field(init=False)
    eval: EvalStage = field(init=False)

    def __post_init__(self) -> None:
        self.workspace.prepare()
        spec = self.spec or {}
        self.data = DataStage(self, spec.get("data"))
        self.train = TrainStage(self, spec.get("train"))
        self.eval = EvalStage(self, spec.get("eval"))

    @property
    def id(self) -> str:
        return self.exp_id

    @property
    def dir(self) -> Path:
        return self.workspace.exp_dir

    @property
    def stages(self) -> list[Any]:
        return [self.data, self.train, self.eval]

    def report(self) -> dict[str, Any]:
        """Recompose ``report.json`` from the stage / artifact summaries and write it."""
        from dataos.report import build_report

        self.updated_at = _utc_now_iso()
        report = build_report(self)
        self.workspace.write_report(report)
        return report


def experiment(
    spec: dict[str, Any] | None = None,
    workspace: str | Path = "workspace",
) -> Experiment:
    """Create a standalone :class:`Experiment` (not attached to a study).

    ``spec`` is an optional dict like ``{"name", "data": {...}, "train": {...},
    "eval": {...}}``; each stage is built from its section. For multi-experiment
    studies use :func:`dataos.study` instead.
    """
    exp_id = _make_id("exp")
    name = (spec or {}).get("name") if spec else None
    return Experiment(
        exp_id=exp_id,
        workspace=Workspace(workspace, exp_id),
        name=str(name) if name else exp_id,
        spec=dict(spec) if spec is not None else None,
    )
