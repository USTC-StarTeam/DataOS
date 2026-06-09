"""Flat experiment report (``report.json``), recomposed from stage summaries.

``build_report`` keeps the historical flat keys (``id`` / ``name`` / ``spec`` /
``status`` / ``dataset`` / ``model`` / ``eval`` ...) so ``Study.summary`` and the
docs keep working, and adds a richer ``stages`` section (each stage's full
``summary()``). Values are derived from the per-stage / per-artifact summaries —
add a field on a stage or artifact and it flows up here automatically.
"""

from __future__ import annotations

from pathlib import Path
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


# -- reproducibility manifest -------------------------------------------------
#
# A per-experiment receipt that records the *recorded* half of reproducibility
# (versions / data revision+fingerprint / config / seeds / hardware / image). The
# *enforced* half is the pinned ``image@digest`` (env.image). Replaying the manifest in
# that image re-runs the experiment — see ``run_one.py --replay``.

_SEED_KEYS = ("seed", "random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed")


def build_manifest(experiment: "Experiment") -> dict[str, Any]:
    """Compose the reproducibility manifest for one experiment (schema ``dataos.manifest/1``)."""
    from dataos import _provenance as prov

    exp_dir = experiment.workspace.exp_dir
    parents = exp_dir.parents
    return {
        "schema": "dataos.manifest/1",
        "who": {
            "name": experiment.name,
            "exp_id": experiment.exp_id,
            "study": parents[1].name if len(parents) > 1 else None,
            "project": parents[2].name if len(parents) > 2 else None,
            "exp_dir": str(exp_dir),
        },
        "created_at": _utc_now_iso(),
        "dataos_version": prov.package_versions().get("dataos"),
        "env": {
            "versions": prov.package_versions(),
            "image": prov.image_ref(),
            "git": prov.git_sha(Path(__file__).parent),
        },
        "hardware": prov.hardware(),
        "data": _data_manifest(experiment.data),
        "train": _train_manifest(experiment.train),
        "eval": _eval_manifest(experiment.eval),
        "result": _result_manifest(experiment),
        "run": _run_command(prov.image_ref()),
    }


def _data_manifest(stage: Any) -> dict[str, Any]:
    art = stage.artifact
    cfg = stage.config or {}
    meta = (art.meta if art is not None else {}) or {}
    return {
        "status": stage.status,
        "source": meta.get("source", cfg.get("source")),
        "revision": meta.get("revision", cfg.get("revision")),
        "fingerprint": meta.get("fingerprint"),
        "rows": meta.get("rows"),
        "columns": meta.get("columns"),
        "recipe": meta.get("recipe", cfg.get("recipe")),
        "config": dict(cfg),  # full spec["data"] (source/split/limit/field_map/recipe/revision) for replay
        "artifact_id": art.id if art is not None else None,
    }


def _train_manifest(stage: Any) -> dict[str, Any]:
    art = stage.artifact
    cfg = dict(stage.config or {})
    meta = (art.meta if art is not None else {}) or {}
    return {
        "status": stage.status,
        "base_model": meta.get("base_model"),
        "template": meta.get("template"),
        "finetuning_type": meta.get("finetuning_type"),
        "mode": meta.get("mode"),
        "seed": cfg.get("seed"),  # recorded only; set it in spec["train"] for reproducible training
        "config": cfg,  # the full open-config (LLaMA-Factory yaml inputs)
        "artifact_id": art.id if art is not None else None,
    }


def _eval_manifest(stage: Any) -> dict[str, Any]:
    art = stage.artifact
    cfg = dict(stage.config or {})
    meta = (art.meta if art is not None else {}) or {}
    return {
        "status": stage.status,
        "backend": meta.get("backend"),
        "mode": meta.get("mode"),
        "tasks": meta.get("tasks", cfg.get("tasks")),
        "num_fewshot": meta.get("num_fewshot", cfg.get("num_fewshot")),
        "seeds": {k: cfg.get(k) for k in _SEED_KEYS if cfg.get(k) is not None},
        "headline_metric": cfg.get("headline_metric"),
        "score": getattr(art, "score", None),
        "metrics": dict(getattr(art, "metrics", {}) or {}),
        "config": cfg,
        "artifact_id": art.id if art is not None else None,
    }


def _result_manifest(experiment: "Experiment") -> dict[str, Any]:
    model = experiment.train.artifact
    evaluation = experiment.eval.artifact
    return {
        "model_id": model.id if model is not None else None,
        "model_path": str(model.path) if model is not None else None,
        "eval_id": evaluation.id if evaluation is not None else None,
        "score": getattr(evaluation, "score", None) if evaluation is not None else None,
        "metrics": dict(getattr(evaluation, "metrics", {}) or {}) if evaluation is not None else {},
    }


def _run_command(image: str | None) -> str:
    img = image or "<image@digest>"
    return (
        f"docker run --rm --gpus '\"device=0\"' -v <workspace>:/ws -e DATAOS_IMAGE={img} {img} "
        "python run-dataos/scripts/run_one.py --replay /ws/<study>/experiments/<name>/manifest.json"
    )
