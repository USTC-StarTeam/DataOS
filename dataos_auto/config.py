"""Run configuration for the autonomous loop.

``Criterion`` / ``Budget`` are the persisted stop-policy (written into project.json
via ``Project.set_goal`` so a killed run can read them back on resume). ``AutoRunConfig``
bundles everything the deterministic skeleton needs for one ``run_auto`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Criterion:
    """Stop condition #1 — bench score target.

    ``metric == "score"`` → the headline ``study.summary()["best"]["score"]``
    (macro-average for a suite). A bench name → that per-task score on the best
    experiment (best-effort; falls back to never-met if unavailable).
    """

    metric: str = "score"
    threshold: float | None = None
    direction: str = "max"  # "max" | "min"

    def as_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "threshold": self.threshold, "direction": self.direction}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Criterion":
        d = d or {}
        return cls(
            metric=d.get("metric", "score"),
            threshold=d.get("threshold"),
            direction=d.get("direction", "max"),
        )


@dataclass
class Budget:
    """Stop conditions #2 and #3 — max rounds and max wall-clock seconds."""

    max_rounds: int | None = None
    max_wall_clock_s: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"max_rounds": self.max_rounds, "max_wall_clock_s": self.max_wall_clock_s}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Budget":
        d = d or {}
        return cls(max_rounds=d.get("max_rounds"), max_wall_clock_s=d.get("max_wall_clock_s"))


@dataclass
class AutoRunConfig:
    """Everything one autonomous run needs.

    The stop policy is (criterion, budget) — at least one of budget's two fields
    should be set so the loop always has a backstop (no unbounded auto-run)."""

    project_id: str
    instruct: str = ""
    goal: str = ""
    scenario: str = "generic"

    criterion: Criterion = field(default_factory=Criterion)
    budget: Budget = field(default_factory=Budget)

    # execution
    gpus: list[str] = field(default_factory=lambda: ["0"])
    workspace: str = "workspace"
    poll_interval_s: float = 10.0
    run_one: str = "run-dataos/scripts/run_one.py"

    # LLM backend
    backend: str = "claude"  # "claude" | "stub"
    model: str = "mimo-v2.5-pro"
    base_url: str = "https://token-plan-cn.xiaomimimo.com/anthropic"
    max_steps: int = 12
    # stub backend only: role ("planner"/"reviewer"/"freeze") -> fixed JSON reply string
    stub_replies: dict[str, str] = field(default_factory=dict)

    # data / eval
    data_source: str | None = None  # hf://... or local path for prepare_once; None → tiny stub data
    data_limit: int | None = None
    eval_cfg: dict | None = None  # pre-frozen eval config; skips discovery when given
    stub: bool = False  # tiny data + stub train/eval, no GPU/HF
