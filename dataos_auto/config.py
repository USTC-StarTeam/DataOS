"""Run configuration for the autonomous loop.

``Criterion`` / ``Budget`` are the persisted stop-policy (written into project.json
via ``Project.set_goal`` so a killed run can read them back on resume). ``AutoRunConfig``
bundles everything the deterministic skeleton needs for one ``run_auto`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PTB_WALL_CLOCK_S = 10 * 3600  # PostTrainBench: 10h / 1 GPU per (base, bench) — the single stop condition


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
    """Wall-clock budget — ``max_wall_clock_s`` is the single PTB stop condition;
    ``max_rounds`` is kept for the record only (NOT a stop condition)."""

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

    The stop policy is a single wall-clock budget (default 10h, the PTB protocol);
    ``criterion`` is only the eval target/metric for ranking + best-over-history,
    NOT a stop condition."""

    project_id: str
    instruct: str = ""
    goal: str = ""
    scenario: str = "generic"

    criterion: Criterion = field(default_factory=Criterion)
    budget: Budget = field(default_factory=lambda: Budget(max_wall_clock_s=PTB_WALL_CLOCK_S))

    # execution
    gpus: list[str] = field(default_factory=lambda: ["0"])
    workspace: str = "workspace"
    poll_interval_s: float = 10.0
    run_one: str = "run-dataos/scripts/run_one.py"

    # LLM backend
    backend: str = "claude_code"  # "claude_code" | "claude" | "openai" | "stub"
    model: str = "mimo-v2.5-pro"  # model id for the claude/openai backends
    base_url: str = "https://token-plan-cn.xiaomimimo.com/anthropic"  # for claude/openai backends
    claude_code_model: str | None = "claude-sonnet-4-6"  # DataOS runs use Sonnet (planner/reviewer/sub-agents)
    max_steps: int = 12
    # stub backend only: role ("planner"/"reviewer"/"freeze") -> fixed JSON reply string
    stub_replies: dict[str, str] = field(default_factory=dict)

    # data / eval
    data_source: str | None = None  # hf://... or local path for prepare_once; None → tiny stub data
    data_limit: int | None = None
    eval_cfg: dict | None = None  # IGNORED on agent-version-ptb (eval is fixed to PostTrainBench)
    stub: bool = False  # tiny data + stub train/eval, no GPU/HF
    ptb_profile: str = "formal"  # PostTrainBench limit profile: "formal" | "smoke"
    target_bench: str | None = None  # PTB single-bench target (per-(base,bench)); None = 7-suite

    # execution: "local" = deterministic run_one (stub/dev); "subagent" = an agentic sub-agent
    # implements each experiment (operators + H100 train/eval) — the real PostTrainBench path.
    train_executor: str = "local"  # "local" | "subagent"
    slurm: dict = field(default_factory=dict)  # H100 connection overrides (runs_base/host/...)
    h100_base_model: str | None = None  # H100 -Base model path (e.g. /data/hcgu/hf_cache/hub/Qwen3-1.7B-Base)
    base_model_id: str | None = None  # logical HF id for the downstream local eval
