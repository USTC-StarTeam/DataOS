"""The three public types and the experiment entry point.

These replace the whole ``dataos.core.artifact`` hierarchy:

* :class:`Model` — a trained model. ``artifact_id`` is a property aliasing
  ``model_id`` and it exposes exactly ``path`` / ``meta`` / ``producer``, so it
  is a drop-in replacement for the old ``ModelArtifact`` everywhere the eval
  subsystem reads a model (no eval function body changes).
* :class:`Eval` — an evaluation result. ``artifact_id`` aliases ``eval_id``.
* :class:`Experiment` — the staged driver. The coding agent calls the stages
  imperatively: ``set_plan`` (record only), ``train``, ``eval``, ``finalize``.
  There is no ``run()`` that chains stages.

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

from dataos.core.task import TaskTemplate
from dataos.workspace import Workspace

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataos_operator import Dataset


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _make_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"{prefix}-{timestamp}-{uuid4().hex[:8]}"


STUB_TASK_DICT: dict[str, Any] = {
    "task_id": "stub",
    "name": "Stub Experiment",
    "goal": "Smoke-test the staged DataOS pipeline without GPU, extras, or an LLM.",
    "input_description": "Any small Alpaca/ShareGPT-shaped dataset.",
    "output_description": "A stub model and a stub eval, plus a flat report.json.",
    "constraints": {},
    "budget": {},
    "max_rounds": 1,
    "train_config": {"mode": "stub"},
    "eval_config": {"provider": "stub"},
}


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
    task: TaskTemplate | None = None
    plan: dict[str, Any] | None = None
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

    # -- stages ---------------------------------------------------------------

    def set_plan(self, plan: dict[str, Any]) -> "Experiment":
        """Record the plan the coding agent decided on (no execution)."""
        self.plan = dict(plan)
        self.status = "planned"
        self._record_step("plan", status="completed", detail={"keys": sorted(self.plan)})
        return self

    def train(self, dataset: "Dataset", config: dict[str, Any] | None = None) -> Model:
        """Train a model from a ``dataos_operator.Dataset`` via the train executor."""
        from dataos.train.llamafactory import run_train

        self.status = "training"
        self.updated_at = _utc_now_iso()
        try:
            model = run_train(
                dataset,
                config,
                workspace=self.workspace,
                task=self.task,
                exp_id=self.exp_id,
            )
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

    def eval(self, model: Model, config: dict[str, Any] | None = None) -> Eval:
        """Evaluate a model via the eval executor."""
        from dataos.eval.runner import run_eval_stage

        self.status = "evaluating"
        self.updated_at = _utc_now_iso()
        try:
            evaluation = run_eval_stage(model, config, workspace=self.workspace)
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
    task: TaskTemplate | dict[str, Any] | None = None,
    workspace: str | Path = "workspace",
) -> Experiment:
    """Create a fresh :class:`Experiment`.

    ``task=None`` falls back to the module-level stub task (``STUB_TASK_DICT``),
    so the no-GPU smoke path needs no template file.
    """
    if task is None:
        resolved_task = TaskTemplate.from_dict(STUB_TASK_DICT)
    elif isinstance(task, TaskTemplate):
        resolved_task = task
    elif isinstance(task, dict):
        resolved_task = TaskTemplate.from_dict(task)
    else:
        raise TypeError(f"task must be a TaskTemplate, dict, or None; got {type(task)!r}")

    exp_id = _make_id("exp")
    return Experiment(exp_id=exp_id, workspace=Workspace(workspace, exp_id), task=resolved_task)
