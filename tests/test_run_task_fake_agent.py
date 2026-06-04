"""Offline end-to-end test: run_task driven by the dataos_agent ``fake`` provider.

No live LLM, no API key, no GPU. The fake agent backend echoes a scripted JSON reply
(injected via ``DATAOS_AGENT_FAKE_REPLY``) that satisfies BOTH the PLAN and REVIEW
Pydantic schemas, so plan and review in the produced report.json are populated by the
agent rather than the old hand-rolled prompt/HTTP/JSON path. Train + eval run in stub
mode (from the task template), producing real model + eval artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from dataos.runtime import run_task

# A single JSON object that validates as PlanModel AND ReviewModel (extra keys ignored).
FAKE_REPLY = json.dumps(
    {
        # PLAN fields
        "steps": [
            {"operator_name": "map_to_alpaca", "params": {"keep_other_columns": False}},
            {"operator_name": "filter_by_text_length", "params": {"min_words": 1, "max_words": 2048}},
            {"operator_name": "select_random_subset", "params": {"max_rows": 3, "seed": 7}},
        ],
        "reasoning_summary": "Normalize to Alpaca, drop degenerate-length rows, then subsample.",
        "expected_gain": "Cleaner, smaller SFT set for a fast stub run.",
        # REVIEW fields
        "summary": "Stub pipeline executed end to end with three data operators.",
        "issues": ["Train and eval are stubbed, so the score is not meaningful."],
        "next_hypothesis": "Swap stub stages for real SFT + eval to measure true gains.",
        "recommended_changes": ["Enable a real train_config mode and eval provider."],
    }
)


def _make_tiny_dataset(path: Path) -> str:
    # ShareGPT-shaped (a `conversations` column), mirroring the bundled datasets so the
    # fixed pre-plan merge accepts it; map_to_alpaca then normalizes it in the plan.
    rows = [
        {
            "conversations": [
                {"from": "human", "value": f"Q{i}: add {i} and {i}"},
                {"from": "gpt", "value": f"The answer is {i + i}."},
            ]
        }
        for i in range(8)
    ]
    Dataset.from_list(rows).save_to_disk(str(path))
    return str(path)


def test_run_task_offline_fake_agent(tmp_path: Path, monkeypatch) -> None:
    # Force the offline fake agent + inject the structured reply via env. No real key.
    monkeypatch.setenv("DATAOS_AGENT_FAKE_REPLY", FAKE_REPLY)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    dataset_dir = _make_tiny_dataset(tmp_path / "tiny_ds")
    task_template = Path("dataos/templates/alpaca_gsm8k_stub_v1.json").resolve()

    result = run_task(
        task_template=task_template,
        datasets=[dataset_dir],
        workspace_dir=str(tmp_path / "workspace"),
        agent_provider="fake",
    )

    assert result.experiment.status == "completed"

    # report.json is durable on disk and carries plan + steps + review.
    report_path = Path(result.experiment.sandbox.report_path())
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # 1) Non-empty plan populated by the fake agent.
    plan = report["plan"]
    assert plan is not None
    assert len(plan["steps"]) == 3
    assert [s["operator_name"] for s in plan["steps"]] == [
        "map_to_alpaca",
        "filter_by_text_length",
        "select_random_subset",
    ]
    assert plan["reasoning_summary"]
    assert plan["expected_gain"]

    # 2) Executed operator steps recorded as completed.
    operator_steps = [s for s in report["steps"] if s["step_type"] == "operator"]
    executed = [s["operator_name"] for s in operator_steps if s["status"] == "completed"]
    for name in ("map_to_alpaca", "filter_by_text_length", "select_random_subset"):
        assert name in executed, f"expected operator {name} to run; got {executed}"

    # 3) A model artifact and an eval artifact exist.
    artifact_types = {a["type"] for a in report["artifacts"]}
    assert "model" in artifact_types, artifact_types
    assert "eval" in artifact_types, artifact_types

    # 4) Review populated by the fake agent (not the fallback text).
    review = report["review"]
    assert review is not None
    assert review["summary"].startswith("Stub pipeline executed end to end")
    assert review["issues"] and review["next_hypothesis"] and review["recommended_changes"]

    # plan.json / review.json also written durably.
    sandbox_root = Path(result.experiment.sandbox.root)
    assert (sandbox_root / "plan.json").exists()
    assert (sandbox_root / "review.json").exists()
