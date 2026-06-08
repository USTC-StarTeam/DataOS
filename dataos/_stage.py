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

from dataos._artifact import Data, Eval, Model
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

    def run(self) -> "Artifact":
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
            raise
        self.status = "completed"
        self.finished_at = _utc_now_iso()
        self.duration_s = round(time.monotonic() - t0, 3)
        self._persist()
        return self.artifact

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
            )
            recipe = cfg.get("recipe") or []
            ds = run_recipe(ds, recipe)
            self.inputs = [str(source)]
        ds.save_to_disk(self.workspace.dataset_dir)
        meta = {
            "source": source,
            "recipe": recipe,
            "format": cfg.get("format"),
            "rows": len(ds),
            "columns": list(ds.hf_dataset.column_names),
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
        if not self.workspace.dataset_dir.exists():
            raise ValueError("train needs data: run exp.data.run() first")
        return Dataset.from_disk(str(self.workspace.dataset_dir))


class EvalStage(Stage):
    """Evaluate the upstream :class:`Model` via the lm-eval bridge."""

    kind = "eval"

    def _run(self) -> Eval:
        from dataos.eval.runner import run_eval_stage

        model = self._pull_model()
        evaluation = run_eval_stage(model, self.config or None, workspace=self.workspace)
        evaluation.inputs = list(self.inputs)
        return evaluation

    def _pull_model(self) -> Model:
        model_art = self.experiment.train.artifact
        if model_art is not None:
            self.inputs = [model_art.id]
            return model_art
        # cross-process: rebuild a ModelLike from the persisted train.json
        train_json = self.workspace.stage_path("train")
        if not train_json.exists():
            raise ValueError("eval needs a model: run exp.train.run() first")
        art = (json.loads(train_json.read_text(encoding="utf-8")) or {}).get("artifact") or {}
        self.inputs = [art["id"]] if art.get("id") else []
        return Model(
            model_id=art.get("id") or "model",
            path=art.get("path") or str(self.workspace.model_dir),
            meta=art.get("meta") or {},
            producer=art.get("producer") or "train",
        )
