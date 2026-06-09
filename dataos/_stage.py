"""Stages — the three first-class steps of an experiment.

A :class:`Stage` is configured (declarative ``config``), executed by ``run()`` (a
template that records status / timing / error and persists itself to
``<exp>/<kind>.json``), and produces an :class:`~dataos._artifact.Artifact`. The
three concrete stages are uniform in shape but differ in who executes:
``DataStage`` runs a deterministic operator recipe (or an injected dataset), while
``TrainStage`` / ``EvalStage`` wrap the LLaMA-Factory / lm-eval executor bodies.

Heavy deps (operators / torch / lm-eval) are imported lazily inside ``run`` so
``import dataos`` stays light.
"""

from __future__ import annotations

import importlib
import json
import time
from typing import TYPE_CHECKING, Any

from dataos._artifact import Data, Eval, Model, artifact_from_dict
from dataos._ids import _make_id, _utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataos._artifact import Artifact
    from dataos._experiment import Experiment


# -- operator recipe executor ------------------------------------------------


def _spec_index() -> dict[str, str]:
    """Map operator name (snake spec name OR class basename) -> dotted class path."""
    from dataos_operator.core import list_operator_specs

    index: dict[str, str] = {}
    for spec in list_operator_specs():
        if not spec.operator_class:
            continue
        index[spec.name] = spec.operator_class
        index[spec.operator_class.rsplit(".", 1)[-1]] = spec.operator_class
    return index


def _import_class(dotted_path: str) -> Any:
    module_path, _, class_name = dotted_path.rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)


def run_recipe(dataset: Any, recipe: list[dict[str, Any]]) -> Any:
    """Apply a declarative recipe ``[{op, params}, ...]`` to a dataset.

    Each step is ``OpClass(**params).run(ds) -> ds`` — uniform: dataset-operators
    auto-wrap a single ``Dataset``, so a linear single-dataset chain needs no
    per-kind dispatch. Operators come from the external ``dataos_operator``,
    resolved by spec name or class basename.
    """
    if not recipe:
        return dataset
    index = _spec_index()
    ds = dataset
    for i, step in enumerate(recipe):
        op_name = step.get("op")
        dotted = index.get(op_name)
        if dotted is None:
            raise ValueError(
                f"recipe[{i}]: unknown operator {op_name!r}; known: {sorted(index)}"
            )
        ds = _import_class(dotted)(**(step.get("params") or {})).run(ds)
    return ds


# -- stages ------------------------------------------------------------------


class Stage:
    """Base stage: declarative config + ``run()`` template + self-recorded state."""

    kind: str = ""

    def __init__(self, experiment: "Experiment", config: dict[str, Any] | None = None) -> None:
        self.experiment = experiment
        self.workspace = experiment.workspace
        self.config: dict[str, Any] = dict(config or {})
        self.status: str = "pending"
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.duration_s: float | None = None
        self.error: str | None = None
        self.log_path: str | None = None
        self.inputs: list[str] = []
        self.artifact: Artifact | None = None

    def run(self, force: bool = False) -> "Artifact":
        # resume: a rehydrated, already-completed stage is skipped, so re-driving a reopened
        # study only runs the pending / failed stages. force=True re-runs unconditionally.
        if not force and self.status == "completed" and self.artifact is not None:
            return self.artifact
        self.status = "running"
        self.started_at = _utc_now_iso()
        t0 = time.monotonic()
        try:
            self.artifact = self._run()
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
            self.finished_at = _utc_now_iso()
            self.duration_s = round(time.monotonic() - t0, 3)
            self._persist()
            self.experiment._persist_report()
            raise
        self.status = "completed"
        self.error = None  # clear any stale error from a prior failed attempt (retry / resume)
        self.finished_at = _utc_now_iso()
        self.duration_s = round(time.monotonic() - t0, 3)
        self._persist()
        self.experiment._persist_report()
        return self.artifact

    def load_state(self) -> bool:
        """Rehydrate this stage from its persisted ``<exp>/<kind>.json`` (reopen / resume).

        Restores status / timing / error / inputs / config / artifact so a reopened study
        reflects prior runs at the object level. Returns True if a state was loaded."""
        path = self.workspace.stage_path(self.kind)
        if not path.is_file():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        self.status = data.get("status") or self.status
        self.started_at = data.get("started_at")
        self.finished_at = data.get("finished_at")
        self.duration_s = data.get("duration_s")
        self.error = data.get("error")
        self.log_path = data.get("log_path")
        self.inputs = list(data.get("inputs") or [])
        if not self.config and data.get("config"):
            self.config = dict(data["config"])
        artifact = data.get("artifact")
        if artifact:
            self.artifact = artifact_from_dict(artifact)
        return True

    def _run(self) -> "Artifact":  # subclass hook — returns an Artifact
        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "error": self.error,
            "log_path": self.log_path,
            "inputs": list(self.inputs),
            "config": self.config,
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
        }

    def _persist(self) -> None:
        path = self.workspace.stage_path(self.kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


class DataStage(Stage):
    """Load a source + run a deterministic operator recipe (or an injected dataset)."""

    kind = "data"

    def __init__(self, experiment: "Experiment", config: dict[str, Any] | None = None) -> None:
        super().__init__(experiment, config)
        self._injected: Any | None = None

    def set(self, dataset: Any) -> "DataStage":
        """Escape hatch: inject a hand-built ``dataos_operator.Dataset`` (skips the recipe)."""
        self._injected = dataset
        return self

    def _run(self) -> Data:
        cfg = self.config
        if self._injected is not None:
            ds = self._injected
            source = cfg.get("source", "<injected>")
            recipe: list[dict[str, Any]] = []
            self.inputs = []
        else:
            source = cfg.get("source")
            if not source:
                raise ValueError(
                    "DataStage needs spec['data']['source'] "
                    "(or inject a dataset via exp.data.set(ds))"
                )
            from dataos.loading import load

            ds = load(
                source,
                split=cfg.get("split", "train"),
                field_map=cfg.get("field_map"),
                limit=cfg.get("limit"),
                revision=cfg.get("revision"),
            )
            recipe = cfg.get("recipe") or []
            ds = run_recipe(ds, recipe)
            self.inputs = [str(source)]
        ds.save_to_disk(self.workspace.dataset_dir)
        meta = {
            "source": source,
            "revision": cfg.get("revision"),
            "recipe": recipe,
            "format": cfg.get("format"),
            "rows": len(ds),
            "columns": list(ds.hf_dataset.column_names),
            # HF's content fingerprint of the materialized dataset — a deterministic
            # hash reused as the dataset's reproducibility id (None if unavailable).
            "fingerprint": getattr(ds.hf_dataset, "_fingerprint", None),
        }
        return Data(_make_id("data"), self.workspace.dataset_dir, meta=meta, inputs=self.inputs)


class TrainStage(Stage):
    """Train a model from the upstream :class:`Data` via the LLaMA-Factory executor."""

    kind = "train"

    def _run(self) -> Model:
        from dataos.train.llamafactory import run_train

        dataset = self._pull_dataset()
        model = run_train(dataset, self.config or None, workspace=self.workspace)
        model.inputs = list(self.inputs)
        return model

    def _pull_dataset(self) -> Any:
        from dataos_operator import Dataset

        data_art = self.experiment.data.artifact
        self.inputs = [data_art.id] if data_art is not None else []
        ddir = self.workspace.dataset_dir
        # dataset_dir is pre-created (empty) by Workspace.prepare(), so .exists() is always True —
        # require an actual save_to_disk payload, else raise the friendly (eval-symmetric) error
        # instead of a cryptic datasets FileNotFoundError.
        if not (ddir / "dataset_info.json").exists() and not (ddir / "dataset_dict.json").exists():
            raise ValueError("train needs data: run exp.data.run() (or exp.data.set(ds).run()) first")
        return Dataset.from_disk(str(ddir))


class EvalStage(Stage):
    """Evaluate the upstream :class:`Model` via the lm-eval bridge."""

    kind = "eval"

    # eval-config keys that name a base model to evaluate directly (no upstream train).
    # They are not EvalConfig fields, so they're consumed here and stripped before the runner.
    _BASELINE_KEYS = ("model_name_or_path", "base_model")

    def _run(self) -> Eval:
        from dataos.eval.runner import run_eval_stage

        model = self._pull_model()
        config = self.config
        if config:
            config = {k: v for k, v in config.items() if k not in self._BASELINE_KEYS} or None
        evaluation = run_eval_stage(model, config, workspace=self.workspace)
        evaluation.inputs = list(self.inputs)
        return evaluation

    def _pull_model(self) -> Model:
        model_art = self.experiment.train.artifact
        if model_art is not None:
            self.inputs = [model_art.id]
            return model_art
        # cross-process: rebuild a ModelLike from the persisted train.json
        train_json = self.workspace.stage_path("train")
        if train_json.exists():
            art = (json.loads(train_json.read_text(encoding="utf-8")) or {}).get("artifact") or {}
            self.inputs = [art["id"]] if art.get("id") else []
            return Model(
                model_id=art.get("id") or "model",
                path=art.get("path") or str(self.workspace.model_dir),
                meta=art.get("meta") or {},
                producer=art.get("producer") or "train",
            )
        # eval-only / baseline: no upstream train — evaluate a bare model named in the eval
        # config (PostTrain "beat the baseline" needs this). The runner turns a non-LoRA Model
        # into pretrained=<path> with no adapter (see eval/backend/native._default_pretrained_source).
        base = self._baseline_model_source()
        if base is not None:
            self.inputs = []
            return Model(
                model_id="baseline",
                path=base,
                meta={"base_model": base, "finetuning_type": None},
                producer="baseline",
            )
        raise ValueError(
            "eval needs a model: run exp.train.run() first, or for an eval-only baseline set "
            "spec['eval']['model_name_or_path'] (or 'base_model', or model_args pretrained=...)"
        )

    def _baseline_model_source(self) -> str | None:
        """A base-model id/path to evaluate when there is no upstream train (PostTrain baseline)."""
        config = self.config or {}
        for key in self._BASELINE_KEYS:
            value = config.get(key)
            if value:
                return str(value)
        model_args = config.get("model_args")
        if isinstance(model_args, dict) and model_args.get("pretrained"):
            return str(model_args["pretrained"])
        if isinstance(model_args, str):
            for part in model_args.split(","):
                part = part.strip()
                if part.startswith("pretrained="):
                    return part[len("pretrained=") :].strip()
        return None
