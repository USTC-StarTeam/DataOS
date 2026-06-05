"""Flat experiment report (``report.json``).

``build_report`` projects an :class:`~dataos.experiment.Experiment` into a flat,
JSON-friendly dict. The coding agent reads this in the review stage and decides
what to do next — DataOS itself makes no judgement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataos._experiment import Experiment


def build_report(experiment: "Experiment") -> dict[str, Any]:
    task = experiment.task
    return {
        "id": experiment.exp_id,
        "status": experiment.status,
        "created_at": experiment.created_at,
        "updated_at": experiment.updated_at,
        "task": _task_section(task),
        "plan": experiment.plan,
        "steps": [dict(step) for step in experiment.steps],
        "dataset": _dataset_section(experiment),
        "model": _model_section(experiment),
        "eval": _eval_section(experiment),
    }


def _task_section(task: Any) -> dict[str, Any] | None:
    if task is None:
        return None
    return {
        "id": getattr(task, "task_id", None),
        "name": getattr(task, "name", None),
        "goal": getattr(task, "goal", None),
    }


def _dataset_section(experiment: "Experiment") -> dict[str, Any] | None:
    dataset = experiment.dataset_info
    if not dataset:
        return None
    return {
        "path": dataset.get("path"),
        "format": dataset.get("format"),
        "rows": dataset.get("rows"),
    }


def _model_section(experiment: "Experiment") -> dict[str, Any] | None:
    model = experiment.model
    if model is None:
        return None
    return {
        "id": model.model_id,
        "path": str(model.path),
        "base_model": model.meta.get("base_model"),
        "finetuning_type": model.meta.get("finetuning_type"),
        "mode": model.meta.get("mode"),
    }


def _eval_section(experiment: "Experiment") -> dict[str, Any] | None:
    evaluation = experiment.eval_result
    if evaluation is None:
        return None
    return {
        "id": evaluation.eval_id,
        "score": evaluation.score,
        "metrics": dict(evaluation.metrics),
        "path": str(evaluation.path),
    }
