"""The three public types and the experiment entry point.

These replace the whole ``dataos.core.artifact`` hierarchy:

* :class:`Model` — a trained model. ``artifact_id`` is a property aliasing
  ``model_id`` and it exposes exactly ``path`` / ``meta`` / ``producer``, so it
  is a drop-in replacement for the old ``ModelArtifact`` everywhere the eval
  subsystem reads a model (no eval function body changes).
* :class:`Eval` — an evaluation result. ``artifact_id`` aliases ``eval_id``.
* :class:`Experiment` — a single staged run. The coding agent calls the stages
  imperatively: ``train``, ``eval``, ``finalize``. There is no ``run()`` that
  chains stages. An experiment carries an optional ``spec`` (``{"name", "train",
  "eval", ...}``) that ``train`` / ``eval`` default their config from; the plan
  that groups multiple experiments lives on :class:`dataos.Study`, not here.

Heavy dependencies (LLaMA-Factory / lm-eval / torch) are imported lazily inside
``train`` / ``eval`` so ``import dataos`` and the stub path work with only
``datasets`` + ``dataos_operator`` installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from dataos.workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataos_operator import Dataset


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _make_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


@dataclass(slots=True)
class Model:
    """A trained model. Drop-in for the old ``ModelArtifact``."""

    model_id: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    producer: str = "train"

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def artifact_id(self) -> str:
        return self.model_id


@dataclass(slots=True)
class Eval:
    """An evaluation result. Drop-in for the old ``EvalArtifact``."""

    eval_id: str
    path: Path
    metrics: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    producer: str = "eval"

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def artifact_id(self) -> str:
        return self.eval_id


@dataclass(slots=True)
class Experiment:
    exp_id: str
    workspace: Workspace
    name: str = ""
    spec: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    dataset_info: dict[str, Any] | None = None
    model: Model | None = None
    eval_result: Eval | None = None
    status: str = "created"
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        self.workspace.prepare()
        self._record_step("create", status="completed")

    @property
    def id(self) -> str:
        return self.exp_id

    @property
    def dir(self) -> Path:
        return self.workspace.exp_dir

    # -- stages ---------------------------------------------------------------

    def train(self, dataset: "Dataset", config: dict[str, Any] | None = None) -> Model:
        """Train a model from a ``dataos_operator.Dataset`` via the train executor.

        ``config`` defaults to ``spec["train"]``; if that is also absent the run
        falls back to stub training (no GPU/torch).
        """
        from dataos.train.llamafactory import run_train

        resolved = config if config is not None else (self.spec or {}).get("train")
        self.status = "training"
        self.updated_at = _utc_now_iso()
        try:
            model = run_train(dataset, resolved, workspace=self.workspace)
        except Exception as exc:
            self._record_step("train", status="failed", detail={"error": str(exc)})
            self.status = "failed"
            self.updated_at = _utc_now_iso()
            raise
        self.model = model
        self.dataset_info = {
            "path": str(self.workspace.dataset_dir),
            "format": model.meta.get("dataset_format"),
            "rows": model.meta.get("selected_rows"),
        }
        self._record_step(
            "train",
            status="completed",
            detail={"model_id": model.model_id, "mode": model.meta.get("mode")},
        )
        self.status = "trained"
        self.updated_at = _utc_now_iso()
        return model

    def eval(self, model: Model | None = None, config: dict[str, Any] | None = None) -> Eval:
        """Evaluate a model via the eval executor.

        ``model`` defaults to this experiment's trained model; ``config`` defaults
        to ``spec["eval"]``. With neither a config nor ``spec["eval"]`` there is
        nothing to evaluate against — pass ``config={"provider": "stub"}`` for the
        no-GPU smoke path.
        """
        from dataos.eval.runner import run_eval_stage

        target = model if model is not None else self.model
        if target is None:
            raise ValueError("eval() needs a model: train one first or pass model=...")
        resolved = config if config is not None else (self.spec or {}).get("eval")
        if resolved is None:
            raise ValueError(
                "eval() needs a config or spec['eval']; "
                "for the no-GPU smoke path pass config={'provider': 'stub'}"
            )
        self.status = "evaluating"
        self.updated_at = _utc_now_iso()
        try:
            evaluation = run_eval_stage(target, resolved, workspace=self.workspace)
        except Exception as exc:
            self._record_step("eval", status="failed", detail={"error": str(exc)})
            self.status = "failed"
            self.updated_at = _utc_now_iso()
            raise
        self.eval_result = evaluation
        self._record_step(
            "eval",
            status="completed",
            detail={"eval_id": evaluation.eval_id, "score": evaluation.score},
        )
        self.status = "evaluated"
        self.updated_at = _utc_now_iso()
        return evaluation

    def finalize(self) -> dict[str, Any]:
        """Write ``report.json`` and return the flat report dict."""
        from dataos.report import build_report

        self.status = "completed"
        self.updated_at = _utc_now_iso()
        self._record_step("finalize", status="completed")
        report = build_report(self)
        self.workspace.write_report(report)
        return report

    # -- helpers --------------------------------------------------------------

    def _record_step(
        self,
        name: str,
        *,
        status: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        step: dict[str, Any] = {
            "name": name,
            "status": status,
            "at": _utc_now_iso(),
        }
        if detail:
            step["detail"] = detail
        self.steps.append(step)


def experiment(
    spec: dict[str, Any] | None = None,
    workspace: str | Path = "workspace",
) -> Experiment:
    """Create a standalone :class:`Experiment` (not attached to a study).

    ``spec`` is an optional dict like ``{"name": ..., "train": {...}, "eval": {...}}``;
    ``train`` / ``eval`` default their config from it. For multi-experiment studies
    use :func:`dataos.study` instead.
    """
    exp_id = _make_id("exp")
    name = (spec or {}).get("name") if spec else None
    return Experiment(
        exp_id=exp_id,
        workspace=Workspace(workspace, exp_id),
        name=str(name) if name else exp_id,
        spec=dict(spec) if spec is not None else None,
    )
