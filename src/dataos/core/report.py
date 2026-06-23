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

from dataos.core.ids import _utc_now_iso
from dataos.core.redact import redact

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataos.core.artifact import Data, Eval, Model
    from dataos.core.experiment import Experiment


def build_report(experiment: "Experiment") -> dict[str, Any]:
    return {
        "id": experiment.exp_id,
        "name": experiment.name,
        "spec": redact(experiment.spec),
        "status": _overall_status(experiment),
        # a headline number is only honest while the eval that produced it is CURRENT —
        # a stale/failed/re-running eval keeps its artifact for forensics, not for ranking
        "score": _report_score(experiment.eval.artifact)
        if experiment.eval.status == "completed"
        else None,
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
    # an invalidated stage anywhere means the chain is NOT current — even if eval still
    # says completed (e.g. eval re-scored before train was rebuilt), don't launder it
    if any(stage.status == "stale" for stage in stages):
        return "stale"
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
        "fingerprint": data.meta.get("fingerprint"),
        "declared_source": data.meta.get("declared_source"),  # injected data: where the spec SAID it came from
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


def _report_score(evaluation: "Eval | None") -> float | None:
    """The experiment's single headline number: the multi-bench macro-average when present,
    else the (single-task) headline score."""
    if evaluation is None:
        return None
    macro = evaluation.meta.get("macro_average")
    return macro if macro is not None else evaluation.score


def _eval_section(evaluation: "Eval | None") -> dict[str, Any] | None:
    if evaluation is None:
        return None
    return {
        "id": evaluation.id,
        "score": evaluation.score,
        "macro_average": evaluation.meta.get("macro_average"),
        "per_task": evaluation.meta.get("per_task"),
        "skipped_tasks": evaluation.meta.get("skipped_tasks"),
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
    from dataos.core import provenance as prov

    import os

    exp_dir = experiment.workspace.exp_dir
    parents = exp_dir.parents
    # an eval-only experiment (baseline / formal confirm by path) never runs data/train;
    # without this flag their "pending" stages make a completed, scored receipt look half-run.
    # Derived from the evaluated model's identity (every eval receipt carries it), so the
    # flag can never disagree with what was actually scored.
    eval_art = experiment.eval.artifact
    eval_only = bool(
        eval_art is not None
        and str((eval_art.meta or {}).get("source_model_id") or "").startswith("evalonly:")
    )
    return {
        "schema": "dataos.manifest/1",
        "eval_only": eval_only,
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
            # exact installed commit of the operator lib (version string alone can't prove
            # which code ran for a git/branch install)
            "dataos_operator": prov.package_origin("dataos-operator"),
        },
        # CUDA_VISIBLE_DEVICES is the device PINNING — hardware() only sees the masked view.
        # Taken from the stage that actually RAN (train, else eval): the manifest writer may
        # be a different process (driver/reopen) whose env says nothing about the run.
        "hardware": {
            **prov.hardware(),
            "visible_devices": (
                getattr(experiment.train, "_visible_devices", None)
                or getattr(experiment.eval, "_visible_devices", None)
                or os.environ.get("CUDA_VISIBLE_DEVICES")
            ),
        },
        "data": _data_manifest(experiment.data),
        "train": _train_manifest(experiment.train),
        "eval": _eval_manifest(experiment.eval),
        "result": _result_manifest(experiment),
        "run": _run_command(prov.image_ref(), exp_dir),
    }


def _data_manifest(stage: Any) -> dict[str, Any]:
    art = stage.artifact
    cfg = stage.config or {}
    meta = (art.meta if art is not None else {}) or {}
    injected = meta.get("source") == "<injected>"
    return {
        "status": stage.status,
        "source": meta.get("source", cfg.get("source")),
        # injected data must not borrow the DECLARED spec's revision/recipe — they describe
        # a pipeline that did not produce these bytes (replay would lie)
        "declared_source": meta.get("declared_source"),
        "revision": meta.get("revision") if injected else meta.get("revision", cfg.get("revision")),
        "fingerprint": meta.get("fingerprint"),
        "rows": meta.get("rows"),
        "columns": meta.get("columns"),
        "recipe": meta.get("recipe") if injected else meta.get("recipe", cfg.get("recipe")),
        "config_drift": getattr(stage, "config_drift", False),
        "config": redact(dict(cfg)),  # full spec["data"] for replay (secret-looking values scrubbed)
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
        "config_drift": getattr(stage, "config_drift", False),  # config differs from what produced the artifact
        "config": redact(cfg),  # the full open-config (LLaMA-Factory yaml inputs), secrets scrubbed
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
        # prefer the effective seeds recorded by the eval runner (EvalConfig defaults included);
        # fall back to user-set seeds in the stage config (e.g. stub eval, which has none).
        "seeds": meta.get("seeds") or {k: cfg.get(k) for k in _SEED_KEYS if cfg.get(k) is not None},
        "headline_metric": cfg.get("headline_metric"),
        # multi-bench / external-harness provenance: what aggregated score this is, which
        # command/harness produced it, and what got skipped — the frozen-eval receipt.
        "macro_average": meta.get("macro_average"),
        "per_task": meta.get("per_task"),
        "command": meta.get("command"),
        "harness": redact(meta.get("harness")),
        "suite": redact(meta.get("suite")),
        "skipped_tasks": redact(meta.get("skipped_tasks")),
        "score": getattr(art, "score", None),
        "metrics": dict(getattr(art, "metrics", {}) or {}),
        "config_drift": getattr(stage, "config_drift", False),
        "config": redact(cfg),
        "artifact_id": art.id if art is not None else None,
    }


def _result_manifest(experiment: "Experiment") -> dict[str, Any]:
    # results are claims about the CURRENT chain — an invalidated (stale) or re-running
    # stage's old artifact stays in its stage receipt for forensics, never in `result`
    model = experiment.train.artifact if experiment.train.status == "completed" else None
    evaluation = experiment.eval.artifact if experiment.eval.status == "completed" else None
    return {
        "model_id": model.id if model is not None else None,
        "model_path": str(model.path) if model is not None else None,
        "eval_id": evaluation.id if evaluation is not None else None,
        "score": getattr(evaluation, "score", None) if evaluation is not None else None,
        "metrics": dict(getattr(evaluation, "metrics", {}) or {}) if evaluation is not None else {},
    }


def _run_command(image: str | None, exp_dir: Path) -> dict[str, str]:
    """How to replay this experiment. Default is **no docker** (uv.lock + manifest suffice);
    docker is an *optional* second form that also pins OS/CUDA."""
    img = image or "<image@digest>"
    return {
        "replay": f"python scripts/run_one.py --replay {exp_dir}/manifest.json --tol 0.3",
        "replay_in_docker": (  # optional: pins the whole environment, not just the python layer
            f"docker run --rm --gpus '\"device=0\"' -v <workspace>:/ws -e DATAOS_IMAGE={img} {img} "
            "python scripts/run_one.py --replay /ws/<study>/experiments/<name>/manifest.json"
        ),
    }
