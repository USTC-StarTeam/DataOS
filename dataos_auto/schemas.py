"""Structured-output contracts for the two LLM judgment points.

These are the Pydantic models passed to ``Agent.run(task, output=...)`` — the
dataos-agent facade injects the JSON schema into the prompt and validates the
model's reply back into a typed instance (``result.output``). No file round-trip,
no emit-tool: the parsing is a first-class facade capability.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExperimentSpec(BaseModel):
    """One experiment in a study plan: the open-config DataOS spec shape.

    ``eval`` is always overwritten by the loop with the frozen shared eval config
    (the planner is not allowed to touch the evaluation protocol)."""

    name: str = Field(description="short unique slug, used as the experiment dir name")
    data: dict = Field(default_factory=dict, description="declarative data spec (MVP: shared, overwritten)")
    train: dict = Field(default_factory=dict, description="LLaMA-Factory pass-through; stub: {'mode':'stub'}")
    eval: dict = Field(default_factory=dict, description="overwritten by the frozen eval config")


class PlanOut(BaseModel):
    """The planner's structured reply: one round's experiment plan."""

    goal: str = Field(description="this round's goal in one sentence")
    experiments: list[ExperimentSpec] = Field(description="experiments to run this round")
    rationale: str = Field(default="", description="why these experiments (audit only, not persisted to plan)")


class ReviewOut(BaseModel):
    """The reviewer's structured reply: conclusion + next-round direction only.

    The reviewer does NOT decide whether to stop — stopping is three hard
    conditions enforced in :mod:`dataos_auto.budget`."""

    conclusion: str = Field(description="qualitative conclusion of this round, grounded in the digest")
    direction: str = Field(default="", description="hint for the next round's planner")
