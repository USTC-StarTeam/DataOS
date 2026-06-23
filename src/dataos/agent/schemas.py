"""Structured-output contracts for the two LLM judgment points.

These are the Pydantic models passed to ``Agent.run(task, output=...)`` — the
dataos-agent facade injects the JSON schema into the prompt and validates the
model's reply back into a typed instance (``result.output``). No file round-trip,
no emit-tool: the parsing is a first-class facade capability.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EdgeSpec(BaseModel):
    """One tree EDGE the planner proposes for an assigned slot: a training recipe
    (data mixture from the fixed pool + SFT hyperparameters). The skeleton fixes which
    node it extends and the experiment name; the base model and eval are frozen by the
    runtime — the planner only fills ``data`` and ``train``."""

    data: dict = Field(default_factory=dict, description="fixed-pool recipe: benchmark_tag/sources/mix/n_samples")
    train: dict = Field(default_factory=dict, description="SFT hparams: lora_rank/lora_alpha/learning_rate/num_train_epochs/...")


class PlanOut(BaseModel):
    """The planner's structured reply: one edge per assigned slot, in slot order."""

    goal: str = Field(description="this round's goal in one sentence")
    edges: list[EdgeSpec] = Field(description="one recipe per slot, in the given slot order")
    rationale: str = Field(default="", description="why these recipes (audit only, not persisted)")


class ReviewOut(BaseModel):
    """The reviewer's structured reply: conclusion + next-round direction only.

    The reviewer does NOT decide whether to stop — stopping is three hard
    conditions enforced in :mod:`dataos.agent.budget`."""

    conclusion: str = Field(description="qualitative conclusion of this round, grounded in the digest")
    direction: str = Field(default="", description="hint for the next round's planner")
