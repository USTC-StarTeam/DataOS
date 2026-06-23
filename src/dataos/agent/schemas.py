"""Structured-output contracts for the agentic main loop.

The main-agent is a Claude Code dialog. It proposes exactly one next experiment
plan at a time, then reviews the completed report. Python owns validation and
state mutation: a JSON reply becomes durable tree state only after it validates
against these models and repository-side invariants.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExperimentPlan(BaseModel):
    """The main-agent's next experiment plan.

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
    expected_signal: str = Field(default="", description="what outcome would be informative")
    subagent_brief: str = Field(
        default="",
        description="concrete instruction for the experiment sub-agent",
    )
    rationale: str = Field(default="", description="why this is the next tree growth")


class ReviewOut(BaseModel):
    """The main-agent's interpretation of one completed node."""

    model_config = ConfigDict(extra="forbid")

    interpretation: str = Field(description="what the report means, grounded in metrics/logs")
    lesson: str = Field(default="", description="what this node teaches the future search")
    recommendation_for_future_growth: str = Field(
        default="",
        description="a concrete hint for later parent/plan choices",
    )
