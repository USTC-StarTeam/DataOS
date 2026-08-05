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
    workspace: str = "workspace"
    poll_interval_s: float = 10.0

    # LLM backend
    backend: str = "cli"  # "cli" (a code-agent CLI via dataos_agent) | "stub"
    agent_provider: str = "kimi"  # dataos_agent registry name: kimi | claude | qwen | ...
    agent_model: str | None = None  # leader and experimenters; None = the CLI's own default model
    agent_effort: str | None = None  # thinking level (claude: low/medium/high/xhigh/max)
    # stall watchdog: kill + resume a CLI agent after Ns of stdout silence (0/None disables)
    agent_idle_timeout_s: float = 1800.0
    max_steps: int = 12
    # stub backend only: "plan"/"review" (or legacy role names) -> fixed JSON reply string(s)
    stub_replies: dict[str, Any] = field(default_factory=dict)

    # data / eval
    eval_cfg: dict | None = None  # IGNORED on agent-version-ptb (eval is fixed to PostTrainBench)
    stub: bool = False  # stub eval score for tests/dev (no GPU/HF)
    ptb_profile: str = "formal"  # PostTrainBench limit profile: "formal" | "smoke"
    eval_limit: int | None = None  # eval size per experiment; -1 = full benchmark (all samples)
    target_bench: str | None = None  # PTB single-bench target (per-(base,bench)); None = 7-suite
    monitor_port: int = 8765  # live-monitor HTTP port (host-side infra fact; 0 disables)

    runtime_config: dict = field(default_factory=dict)  # cluster paths/envs injected into experimenters
    runtime_config_path: str | None = None  # provenance for runtime_config when loaded from disk
    h100_base_model: str | None = None  # H100 -Base model path (e.g. /data/hcgu/hf_cache/hub/Qwen3-1.7B-Base)
    base_model_id: str | None = None  # logical HF id for the downstream local eval
