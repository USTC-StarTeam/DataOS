"""Experiment — a single staged run composed of three Stage objects.

An experiment is a container: it owns a workspace directory and composes a
:class:`~dataos._stage.DataStage` / :class:`~dataos._stage.TrainStage` /
:class:`~dataos._stage.EvalStage` (built from its ``spec``). The coding agent
drives the stages imperatively — ``exp.data.run()`` -> ``exp.train.run()`` ->
``exp.eval.run()`` — there is no ``run()`` that chains them. ``summary()`` writes a
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

    def summary(self) -> dict[str, Any]:
        """Return this experiment's current situation, composed from the stage / artifact
        summaries. **Pure** — no side effects; call it after a stage (or any time) to
        inspect. ``report.json`` is persisted automatically by each :meth:`Stage.run`
        (via :meth:`_persist_report`), so a fully-run experiment is visible to
        ``Study.summary`` whether or not you ever call this."""
        from dataos.report import build_report

        return build_report(self)

    def _persist_report(self) -> None:
        """Refresh ``report.json`` + ``manifest.json`` from current state. Called by each
        ``Stage.run`` so both always reflect the latest stage state (progress included)."""
        from dataos.report import build_manifest

        self.updated_at = _utc_now_iso()
        self.workspace.write_report(self.summary())
        self.workspace.write_manifest(build_manifest(self))

    def load_state(self) -> "Experiment":
        """Rehydrate all three stages from their persisted ``{kind}.json`` (reopen / resume)."""
        for stage in self.stages:
            stage.load_state()
        return self


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
