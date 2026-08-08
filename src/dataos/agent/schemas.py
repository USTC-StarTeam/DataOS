"""Structured-output contracts for the agentic main loop.

The leader is a Claude Code dialog. It proposes exactly one next experiment
plan at a time, then reviews the completed report. Python owns validation and
state mutation: a JSON reply becomes durable tree state only after it validates
against these models and repository-side invariants.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExperimentPlan(BaseModel):
    """The leader's next experiment plan.

    The agent chooses a parent node and executable data/train knobs. Code generates
    the child node id, injects the parent model path, freezes eval, and writes state.
    """

    model_config = ConfigDict(extra="forbid")

    parent_node_id: str = Field(description="existing tree node to continue training from")
    name: str = Field(description="short experiment name, unique within this project")
    hypothesis: str = Field(description="what this child experiment is testing")
    estimated_hours: float | None = Field(
        default=None,
        description="rough wall-clock estimate for the full data/train/eval experiment",
    )
    data: dict = Field(
        default_factory=dict,
        description="executable fixed-pool data recipe: sources plus operator pipeline",
    )
    train: dict = Field(
        default_factory=dict,
        description="SFT hyperparameters; do not include model_name_or_path",
    )
    eval_plan: str = Field(
        default="",
        description="short note about what signal to read from the frozen evaluator",
    )
    eval_limit: int | None = Field(
        default=None,
        description="eval size for THIS experiment: -1 = full frozen benchmark, or a "
        "positive subset size. REQUIRED when the run's eval mode is auto (-2) — pick by "
        "the experiment's role: exploration probe -> a subset sized to the bench's eval "
        "cost; a candidate-best confirmation -> -1 (full). Ignored in fixed-limit runs.",
    )
    expected_signal: str = Field(default="", description="what outcome would be informative")
    subagent_brief: str = Field(
        default="",
        description="concrete instruction for the experiment experimenter",
    )
    rationale: str = Field(default="", description="why this is the next tree growth")


class ReviewOut(BaseModel):
    """The leader's interpretation of one completed node."""

    model_config = ConfigDict(extra="forbid")

    interpretation: str = Field(description="what the report means, grounded in metrics/logs")
    lesson: str = Field(default="", description="what this node teaches the future search")
    recommendation_for_future_growth: str = Field(
        default="",
        description="a concrete hint for later parent/plan choices",
    )
