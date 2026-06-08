"""Artifacts — the uniform leaf outputs of stages, with lineage.

Every stage produces an :class:`Artifact`: :class:`Data` (dataset), :class:`Model`,
:class:`Eval`. They share ``id`` / ``kind`` / ``path`` / ``meta`` / ``producer`` /
``inputs`` (upstream artifact ids = lineage) / ``created_at``, a ``to_dict`` (full
record) and a ``summary`` (compact view for report / review).

``Model`` / ``Eval`` keep their historical constructors (``Model(model_id=...)``,
``Eval(eval_id=...)``) and ``.model_id`` / ``.eval_id`` aliases, so the train / eval
executor bodies and the eval engine — which reads only ``.path`` / ``.meta`` /
``.artifact_id`` / ``.producer`` via the ``ModelLike`` protocol — need no changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dataos._ids import _utc_now_iso


@dataclass
class Artifact:
    """A uniform immutable leaf output with provenance + lineage."""

    id: str
    kind: str  # "data" | "model" | "eval"
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    producer: str = ""
    inputs: list[str] = field(default_factory=list)  # upstream artifact ids (lineage)
    created_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def artifact_id(self) -> str:
        return self.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "path": str(self.path),
            "producer": self.producer,
            "inputs": list(self.inputs),
            "created_at": self.created_at,
            "meta": dict(self.meta),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "path": str(self.path),
            "producer": self.producer,
            "inputs": list(self.inputs),
        }


class Data(Artifact):
    """A prepared training dataset (materialized to ``path`` via ``save_to_disk``)."""

    def __init__(
        self,
        data_id: str,
        path: Path | str,
        meta: dict[str, Any] | None = None,
        producer: str = "data",
        inputs: list[str] | None = None,
    ) -> None:
        super().__init__(
            id=data_id, kind="data", path=path, meta=meta or {},
            producer=producer, inputs=list(inputs or []),
        )

    @property
    def data_id(self) -> str:
        return self.id

    def load(self) -> Any:
        """Reload the live ``dataos_operator.Dataset`` from disk."""
        from dataos_operator import Dataset

        return Dataset.from_disk(str(self.path))

    def summary(self) -> dict[str, Any]:
        return {
            **super().summary(),
            "format": self.meta.get("format"),
            "rows": self.meta.get("rows"),
            "columns": self.meta.get("columns"),
        }


class Model(Artifact):
    """A trained model. Drop-in for the old ``ModelArtifact`` (satisfies ModelLike)."""

    def __init__(
        self,
        model_id: str,
        path: Path | str,
        meta: dict[str, Any] | None = None,
        producer: str = "train",
        inputs: list[str] | None = None,
    ) -> None:
        super().__init__(
            id=model_id, kind="model", path=path, meta=meta or {},
            producer=producer, inputs=list(inputs or []),
        )

    @property
    def model_id(self) -> str:
        return self.id

    def summary(self) -> dict[str, Any]:
        return {
            **super().summary(),
            "base_model": self.meta.get("base_model"),
            "template": self.meta.get("template"),
            "finetuning_type": self.meta.get("finetuning_type"),
            "mode": self.meta.get("mode"),
        }


class Eval(Artifact):
    """An evaluation result."""

    def __init__(
        self,
        eval_id: str,
        path: Path | str,
        metrics: dict[str, Any] | None = None,
        score: float | None = None,
        meta: dict[str, Any] | None = None,
        producer: str = "eval",
        inputs: list[str] | None = None,
    ) -> None:
        super().__init__(
            id=eval_id, kind="eval", path=path, meta=meta or {},
            producer=producer, inputs=list(inputs or []),
        )
        self.metrics: dict[str, Any] = dict(metrics or {})
        self.score: float | None = score

    @property
    def eval_id(self) -> str:
        return self.id

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "metrics": dict(self.metrics), "score": self.score}

    def summary(self) -> dict[str, Any]:
        return {**super().summary(), "score": self.score, "metrics": dict(self.metrics)}


def artifact_from_dict(data: dict[str, Any]) -> Artifact:
    """Rebuild a :class:`Data` / :class:`Model` / :class:`Eval` from a persisted ``to_dict``
    payload — used to rehydrate a stage's artifact when a Study is reopened from disk."""
    kind = data.get("kind")
    art_id = data.get("id") or ""
    path = data.get("path") or ""
    meta = dict(data.get("meta") or {})
    producer = data.get("producer") or ""
    inputs = list(data.get("inputs") or [])
    if kind == "data":
        artifact: Artifact = Data(art_id, path, meta=meta, producer=producer, inputs=inputs)
    elif kind == "model":
        artifact = Model(model_id=art_id, path=path, meta=meta, producer=producer, inputs=inputs)
    elif kind == "eval":
        artifact = Eval(
            eval_id=art_id, path=path, metrics=dict(data.get("metrics") or {}),
            score=data.get("score"), meta=meta, producer=producer, inputs=inputs,
        )
    else:
        artifact = Artifact(
            id=art_id, kind=str(kind or ""), path=path, meta=meta, producer=producer, inputs=inputs
        )
    if data.get("created_at"):
        artifact.created_at = data["created_at"]
    return artifact
