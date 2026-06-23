"""L1 smoke: real LLM returns a parseable ExperimentPlan.

Skipped unless a key (MIMO_API_KEY or ANTHROPIC_API_KEY) is set AND the `claude` CLI is
on PATH. No GPU and no training — it only proves the structured-output contract is stable.
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
def test_main_agent_real_llm_plan(tmp_path):
    from dataos.agent.backend import make_agent
    from dataos.agent.config import AutoRunConfig, Budget, Criterion
    from dataos.agent.planner import next_plan
    from dataos.agent.schemas import ExperimentPlan

    cfg = AutoRunConfig(
        project_id="l1",
        workspace=str(tmp_path / "ws"),
        backend="claude",
        goal="raise gsm8k accuracy of a small base model",
        instruct="fine-tune a base model to improve gsm8k",
        criterion=Criterion(threshold=0.5),
        budget=Budget(max_rounds=1),
    )
    context = {
        "setting": {"goal": cfg.goal, "target_bench": "gsm8k"},
        "budget": {"remaining_hours": 10},
        "tree": {
            "growable_node_ids": ["n0000"],
            "nodes": [{"node_id": "n0000", "status": "completed", "has_model": True}],
        },
        "constraints": ["parent_node_id must be n0000"],
    }
    plan, _ = next_plan(agent=make_agent("main_agent", cfg), context=context, catalog=[])
    assert isinstance(plan, ExperimentPlan)
    assert plan.parent_node_id == "n0000"
    assert plan.name
