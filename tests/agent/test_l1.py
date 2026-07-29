"""L1 smoke: the main-agent (cli backend) returns a parseable ExperimentPlan.

Skipped unless the configured provider's CLI is on PATH. The cli backend (dataos_agent)
uses the CLI's own local login — no API key from config. No GPU and no training; it only
proves the structured-output contract.
"""

import shutil

import pytest

from dataos.agent.config import AutoRunConfig

_HAS_CLI = shutil.which(AutoRunConfig(project_id="probe").agent_provider) is not None


@pytest.mark.skipif(
    not _HAS_CLI,
    reason="needs the configured agent_provider CLI on PATH (cli backend uses its login)",
)
def test_main_agent_real_llm_plan(tmp_path):
    from dataos.agent.backend import make_agent
    from dataos.agent.config import Budget, Criterion
    from dataos.agent.planner import next_plan
    from dataos.agent.schemas import ExperimentPlan

    cfg = AutoRunConfig(
        project_id="l1",
        workspace=str(tmp_path / "ws"),
        backend="cli",
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
