"""The agent-facing ``ModelJobSpec`` and its interaction union.

A model job is the layer-2 execution slot of the DataOS operator stack: it scores data
WITH a model (one forward pass, no training) so offline operators can consume per-sample
signals downstream. It is a peer of ``TrainSpec`` — same lifecycle (validate / build /
submit / status / collect), same reused building blocks (``DatasetRef``, ``Resources``,
``EnvSpec``, ``cluster``) — but emits ``scores.jsonl`` instead of a checkpoint.
"""

from __future__ import annotations

from typing import Annotated, Literal

from dataos_launch.model import Model
from dataos_launch.types import Resources
from pydantic import Field

from ..spec import DatasetRef, EnvSpec


class LossInteraction(Model):
    """Per-sample loss scoring over the final assistant segment of each record.

    ``output_key`` is the field convention the score is written back to when the driving
    operator merges ``scores.jsonl`` into the dataset (offline ``.run()`` reads it, e.g.
    ``LossSelector(loss_key="metadata.runtime.loss")``).
    """

    kind: Literal["loss"] = "loss"
    batch_size: int = Field(default=4, gt=0)
    max_len: int = Field(default=512, gt=0)
    output_key: str = "metadata.runtime.loss"


class GradientInteraction(Model):
    """LESS-style gradient-feature scoring (arXiv 2402.04333).

    Each record gets a low-dimensional feature: the completion-only CE gradient w.r.t. a
    temporary UNTRAINED LoRA adapter (base model frozen), sign-normalized, then projected
    to ``projection_dim`` by a Rademacher matrix derived from ``projection_seed``. The
    seed is shared by ALL candidate and target records (one projection matrix) — a hard
    constraint: per-record seeds make cosine similarities incomparable (historical bug).

    With ``target`` given (e.g. a bench train split), ``score`` is the cosine similarity
    to the mean normalized target feature; without it, ``score`` is the projected
    vector's L2 norm. ``output_key`` is the field convention consumed downstream by
    offline selectors (``LossSelector``/``LessSelector``).
    """

    kind: Literal["gradient"]
    projection_dim: int = Field(default=1024, gt=0)
    projection_seed: int = 42
    lora_r: int = Field(default=8, gt=0)
    max_len: int = Field(default=512, gt=0)
    target: DatasetRef | None = None
    output_key: str = "metadata.less.score"


Interaction = Annotated[LossInteraction | GradientInteraction, Field(discriminator="kind")]


class ModelJobSpec(Model):
    """Facts for one scoring job. ``model`` is injected by the runtime, never the LLM."""

    name: str
    model: str  # resolved HF model dir, injected by DataOS
    dataset: DatasetRef
    interaction: Interaction = Field(default_factory=LossInteraction)
    resources: Resources = Field(default_factory=Resources)
    env: EnvSpec = Field(default_factory=EnvSpec)
