"""Flat experiment report (``report.json``), recomposed from stage summaries.

``build_report`` keeps the historical flat keys (``id`` / ``name`` / ``spec`` /
``status`` / ``dataset`` / ``model`` / ``eval`` ...) so ``Study.summary`` and the
docs keep working, and adds a richer ``stages`` section (each stage's full
``summary()``). Values are derived from the per-stage / per-artifact summaries —
add a field on a stage or artifact and it flows up here automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataos._ids import _utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataos._artifact import Data, Eval, Model
    from dataos._experiment import Experiment


def build_report(experiment: "Experiment") -> dict[str, Any]:
    return {
        "id": experiment.exp_id,
        "name": experiment.name,
        "spec": experiment.spec,
        "status": _overall_status(experiment),
        "created_at": experiment.created_at,
        "updated_at": _utc_now_iso(),
        "dataset": _dataset_section(experiment.data.artifact),
        "model": _model_section(experiment.train.artifact),
        "eval": _eval_section(experiment.eval.artifact),
        "stages": {stage.kind: stage.summary() for stage in experiment.stages},
    }


def _overall_status(experiment: "Experiment") -> str:
    stages = experiment.stages
    if any(stage.status == "failed" for stage in stages):
        return "failed"
    if experiment.eval.status == "completed":
        return "completed"
    done = [stage.kind for stage in stages if stage.status == "completed"]
    return done[-1] if done else "created"


def _dataset_section(data: "Data | None") -> dict[str, Any] | None:
    if data is None:
        return None
    return {
        "path": str(data.path),
        "format": data.meta.get("format"),
        "rows": data.meta.get("rows"),
    }


def _model_section(model: "Model | None") -> dict[str, Any] | None:
    if model is None:
        return None
    return {
        "id": model.id,
        "path": str(model.path),
        "base_model": model.meta.get("base_model"),
        "finetuning_type": model.meta.get("finetuning_type"),
        "mode": model.meta.get("mode"),
    }


def _eval_section(evaluation: "Eval | None") -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "id": evaluation.id,
        "score": evaluation.score,
        "metrics": dict(evaluation.metrics),
        "path": str(evaluation.path),
    }
