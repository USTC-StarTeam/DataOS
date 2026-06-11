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
import os
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dataos._artifact import Data, Eval, Model, artifact_from_dict
from dataos._io import atomic_write_json
from dataos._redact import redact
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
        self.config_drift: bool = False  # reopened with a config that differs from the persisted one
        self._runner: dict[str, Any] = {}  # liveness identity while running: {pid, hostname}
        self._visible_devices: str | None = None  # CUDA pinning of the process that RAN this stage

    def run(self, force: bool = False) -> "Artifact":
        # liveness guard: a rehydrated "running" stage may STILL be running in another
        # process — re-driving it would double-run into the same experiment dirs. Re-read
        # the persisted state first: the in-memory snapshot is load-time-stale (TOCTOU).
        self._refresh_runner_from_disk()
        if not force and self.status == "running" and self._runner_alive():
            raise RuntimeError(
                f"{self.kind} stage is already running (pid {self._runner.get('pid')} on "
                f"{self._runner.get('hostname')}) — re-driving would double-run this experiment "
                "dir. Wait for it to finish, or run(force=True) after confirming it is dead."
            )
        # resume: a rehydrated, already-completed stage is skipped, so re-driving a reopened
        # study only runs the pending / failed / stale ones. A completed stage whose config
        # DRIFTED from the persisted receipt must NOT be skipped: keeping the old artifact
        # while reporting the new config would make the receipts lie. force=True always re-runs.
        if (
            not force
            and self.status == "completed"
            and self.artifact is not None
            and not self.config_drift
        ):
            return self.artifact
        self.status = "running"
        self.started_at = _utc_now_iso()
        self._visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self._runner = {"pid": os.getpid(), "hostname": socket.gethostname()}
        # the old artifact's files are about to be wiped — receipts must stop advertising it
        # NOW, not keep presenting a dead artifact id/score as live through the run/failure
        self.artifact = None
        # persist the running state immediately: a 40-min train/eval must be observable from
        # {kind}.json (pollers had to grep tqdm logs when this was only written at stage end)
        self._persist()
        t0 = time.monotonic()
        try:
            # no stale files from a prior run/config (e.g. lora→full retrain leaving
            # adapter_config.json behind, which eval would then load as a stale adapter).
            # Exception: a train resuming from its own checkpoints must keep them.
            if not (self.kind == "train" and (self.config or {}).get("resume_from_checkpoint")):
                self.workspace.clear_artifact_dir(self.kind)
            self.artifact = self._run()
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
            self.finished_at = _utc_now_iso()
            self.duration_s = round(time.monotonic() - t0, 3)
            self._runner = {}
            self._persist()
            self.experiment._persist_report()
            raise
        self.status = "completed"
        self.error = None  # clear any stale error from a prior failed attempt (retry / resume)
        self.config_drift = False
        self.finished_at = _utc_now_iso()
        self.duration_s = round(time.monotonic() - t0, 3)
        self._runner = {}
        self._persist()
        # this stage actually ran -> downstream completed artifacts were built from
        # now-stale upstream outputs; mark them stale so the next drive re-runs them
        self.experiment._invalidate_downstream(self.kind)
        self.experiment._persist_report()
        return self.artifact

    def _runner_alive(self) -> bool:
        pid, host = self._runner.get("pid"), self._runner.get("hostname")
        if not pid or host != socket.gethostname():
            return False  # cross-host liveness is unknowable -> treat as stale, proceed
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # EPERM = a LIVE process owned by someone else (shared box) — alive
        except (OSError, ValueError):
            return False
        return True

    def _refresh_runner_from_disk(self) -> None:
        """Pull the latest persisted status/runner before the liveness check — another
        process may have started (or finished) this stage since we rehydrated."""
        path = self.workspace.stage_path(self.kind)
        if not path.is_file():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return  # torn read mid-replace — keep the in-memory view
        persisted_status = data.get("status")
        if persisted_status == "running":
            self.status = "running"
            self._runner = dict(data.get("runner") or {})

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
        self._runner = dict(data.get("runner") or {})
        self._visible_devices = data.get("visible_devices")
        persisted_config = data.get("config")
        if not self.config and persisted_config:
            self.config = dict(persisted_config)
        elif persisted_config is not None and self.config and self.status == "completed":
            # the persisted copy is redacted — compare like-for-like. A completed stage
            # reopened with a DIFFERENT config must not silently keep its old artifact
            # while the receipts start claiming the new config produced it.
            current = json.dumps(redact(self.config), sort_keys=True, default=str)
            persisted = json.dumps(persisted_config, sort_keys=True, default=str)
            self.config_drift = current != persisted
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
            "config": redact(self.config),  # persisted copies never carry tokens (env is the channel)
            "config_drift": self.config_drift,
            "runner": self._runner or None,  # liveness identity while running ({pid, hostname})
            "visible_devices": self._visible_devices,  # device pinning of the RUNNING process
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
        }

    def _persist(self) -> None:
        atomic_write_json(self.workspace.stage_path(self.kind), self.summary())


class DataStage(Stage):
    """Load a source + run a deterministic operator recipe (or an injected dataset)."""

    kind = "data"

    def __init__(self, experiment: "Experiment", config: dict[str, Any] | None = None) -> None:
        super().__init__(experiment, config)
        self._injected: Any | None = None

    def run(self, force: bool = False) -> "Artifact":
        # spec-config drift can't see injected datasets (their identity lives in the object,
        # not the spec) — compare content fingerprints so re-driving a completed experiment
        # with a DIFFERENT prepared pool re-runs instead of silently keeping the old data
        if (
            self._injected is not None
            and self.status == "completed"
            and self.artifact is not None
        ):
            new_fp = getattr(getattr(self._injected, "hf_dataset", None), "_fingerprint", None)
            old_fp = (self.artifact.meta or {}).get("fingerprint")
            if new_fp is not None and old_fp is not None and new_fp != old_fp:
                self.config_drift = True
        return super().run(force)

    def set(self, dataset: Any) -> "DataStage":
        """Escape hatch: inject a hand-built ``dataos_operator.Dataset`` (skips the recipe)."""
        self._injected = dataset
        return self

    def _run(self) -> Data:
        cfg = self.config
        if self._injected is not None:
            ds = self._injected
            # honest provenance: whatever spec["data"] declares, THIS dataset came from
            # set() — recording the declared source would launder injected data as
            # declarative and make --replay re-run a recipe that never produced it.
            source = "<injected>"
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
            "declared_source": cfg.get("source") if self._injected is not None else None,
            "revision": cfg.get("revision"),
            "recipe": recipe,
            # injected datasets carry their format in operator metadata, not the spec
            "format": cfg.get("format") or (getattr(ds, "metadata", None) or {}).get("dataset_format"),
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
        train_status = self.experiment.train.status
        model_art = self.experiment.train.artifact
        if model_art is not None:
            # an artifact whose stage is stale/failed/running is NOT scoreable: evaluating
            # it launders an invalidated chain back into a "completed" report
            if train_status != "completed":
                raise ValueError(
                    f"upstream train stage is {train_status!r} — its model is not current; "
                    "run exp.train.run() first (or run(force=True) to rebuild the chain)"
                )
            self.inputs = [model_art.id]
            return model_art
        # cross-process: rebuild a ModelLike from the persisted train.json
        train_json = self.workspace.stage_path("train")
        if train_json.exists():
            persisted = json.loads(train_json.read_text(encoding="utf-8")) or {}
            art = persisted.get("artifact") or {}
            if art:
                if persisted.get("status") != "completed":
                    raise ValueError(
                        f"persisted train stage is {persisted.get('status')!r} — its model is "
                        "not current; re-run training before evaluating"
                    )
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
            # identity must come from WHAT is being evaluated, not from an assumed role:
            # this path also scores trained models referenced by path (formal confirms),
            # and labeling those "baseline" made the deciding receipts misname the candidates.
            # basename alone is ambiguous (every merged model ends in ".../_merged") — add a
            # short hash of the FULL reference so two candidates can't share an identity.
            import hashlib

            name = Path(str(base)).name or str(base)
            digest = hashlib.sha1(str(base).encode("utf-8")).hexdigest()[:8]
            return Model(
                model_id=f"evalonly:{name}#{digest}",
                path=base,
                meta={"base_model": base, "finetuning_type": None, "eval_only": True},
                producer="eval_only",
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
