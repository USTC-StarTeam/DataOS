"""Shared runtime dataclasses for policy-driven operator execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from dataos_operator.core.contracts import Record


RuntimeMode = Literal["offline", "online", "hybrid"]


@dataclass
class ArtifactRef:
    """Describes a runtime artifact produced or consumed by an operator."""

    name: str
    kind: str
    path: str | None = None
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorState:
    """Mutable state container owned by one operator instance."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ArtifactRef] = field(default_factory=list)


@dataclass
class RuntimeContext:
    """Execution context shared by policy-aware operators."""

    mode: RuntimeMode = "offline"
    seed: int = 42
    step: int = 0
    epoch: float = 0.0
    update_interval: int = 1
    trainer_backend: str | None = None
    artifact_dir: str | None = None
    cache_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchView:
    """The current batch presented to runtime-aware operators."""

    records: list[Record]
    indices: list[int]
    domain_ids: list[str] | None = None
    tensors: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForwardSignal:
    """Signals emitted after forward propagation or offline scoring."""

    per_sample_losses: list[float] | None = None
    outputs: Any = None
    embeddings: list[list[float]] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackwardSignal:
    """Signals emitted after backward propagation."""

    gradient_vector: list[float] | None = None
    gradient_norms: list[float] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyAction:
    """Action proposed by one operator at any runtime phase."""

    selected_indices: list[int] | None = None
    sample_weights: list[float] | None = None
    domain_weights: dict[str, float] | None = None
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
