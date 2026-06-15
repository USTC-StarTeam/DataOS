"""L1 smoke: real LLM (claude backend / MiMo) returns a parseable PlanOut.

Skipped unless a key (MIMO_API_KEY or ANTHROPIC_API_KEY) is set AND the `claude` CLI is
on PATH. No GPU and no training — it only proves the structured-output contract is stable
on the real Anthropic-compatible endpoint. Run on the a800 box where the MiMo key lives.
"""

import os
import shutil

import pytest

_HAS_KEY = bool(os.environ.get("MIMO_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
_HAS_CLI = shutil.which("claude") is not None


@pytest.mark.skipif(
    not (_HAS_KEY and _HAS_CLI),
    reason="needs MIMO_API_KEY/ANTHROPIC_API_KEY + the claude CLI on PATH",
)
def test_planner_real_llm(tmp_path):
    from dataos_auto.backend import make_agent
    from dataos_auto.config import AutoRunConfig, Budget, Criterion
    from dataos_auto.planner import next_plan
    from dataos_auto.schemas import PlanOut

    cfg = AutoRunConfig(
        project_id="l1",
        workspace=str(tmp_path / "ws"),
        backend="claude",
        goal="raise gsm8k accuracy of a small base model",
        instruct="fine-tune a base model to improve gsm8k",
        criterion=Criterion(threshold=0.5),
        budget=Budget(max_rounds=1),
    )
    plan = next_plan(
        agent=make_agent("planner", cfg),
        instruct=cfg.instruct,
        goal=cfg.goal,
        eval_cfg={"tasks": ["gsm8k"]},
        round_idx=0,
        catalog=[],
    )
    assert isinstance(plan, PlanOut)
    assert plan.experiments and all(e.name for e in plan.experiments)
