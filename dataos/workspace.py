"""Flat per-experiment workspace.

Replaces the old ``dataos.core.sandbox.Sandbox``. Each experiment lives in a
single flat directory::

    <workspace>/<exp_id>/
        dataset/        # materialized training Dataset (save_to_disk)
        model/          # trained model artifact (weights / adapter)
        eval/           # eval artifact (results, protocol, samples)
        logs/           # human/agent-readable step logs
        _lf/            # LLaMA-Factory scratch (exported data + train.yaml)
        report.json     # flat experiment report

The method/property names mirror the surface the eval runner and the ported
train helpers already call (``artifact_dir``, ``log``, ``tmp_dir``,
``code_dir``, ``logs_dir``) so those bodies do not change. ``artifact_dir``
maps ``kind="model"`` to ``model_dir`` and ``kind="eval"`` to ``eval_dir`` on
the flat layout (there is no per-id nesting any more — one model + one eval per
experiment).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Workspace:
    root: Path | str
    exp_id: str

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def exp_dir(self) -> Path:
        return self.root / self.exp_id

    @property
    def dataset_dir(self) -> Path:
        return self.exp_dir / "dataset"

    @property
    def model_dir(self) -> Path:
        return self.exp_dir / "model"

    @property
    def eval_dir(self) -> Path:
        return self.exp_dir / "eval"

    @property
    def logs_dir(self) -> Path:
        return self.exp_dir / "logs"

    @property
    def tmp_dir(self) -> Path:
        return self.exp_dir / "_lf"

    @property
    def code_dir(self) -> Path:
        return self.exp_dir / "_lf"

    def prepare(self) -> None:
        for path in (
            self.exp_dir,
            self.dataset_dir,
            self.model_dir,
            self.eval_dir,
            self.logs_dir,
            self.tmp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def artifact_dir(self, kind: str, _id: str) -> Path:
        del _id  # flat layout keeps a single dir per kind
        if kind == "model":
            target = self.model_dir
        elif kind == "eval":
            target = self.eval_dir
        elif kind == "dataset":
            target = self.dataset_dir
        else:
            raise ValueError(f"Unsupported workspace artifact kind: {kind!r}")
        target.mkdir(parents=True, exist_ok=True)
        return target

    def log(self, name: str, content: str) -> Path:
        filename = name if name.endswith(".log") else f"{name}.log"
        target = self.logs_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def report_path(self) -> Path:
        return self.exp_dir / "report.json"

    def write_report(self, data: dict[str, Any]) -> Path:
        target = self.report_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return target
