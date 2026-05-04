"""Integration test: NL goal -> plan -> execute through lm-eval bridge.

Uses a fake adapter so this stays CPU-only (no GPU, no large model). Verifies
that ``plan_eval`` + ``execute_plan`` end-to-end produce a sane PlanRunReport
on a real lm-eval task (gsm8k). Skipped if lm-eval is not installed.
"""
from __future__ import annotations

import os

import pytest

from dataos.core.artifact import ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.eval.planner import EvalPlanItem, execute_plan, plan_eval
from dataos.eval.types import EvalConfig, EvalRequest, ModelResponse

pytest.importorskip("lm_eval")


class _FakeAdapter:
    name = "fake"

    def generate_until(
        self,
        model_artifact: ModelArtifact,
        requests: list[EvalRequest],
        config: EvalConfig,
    ) -> list[ModelResponse]:
        return [
            ModelResponse(
                sample_id=request.sample_id,
                index=request.index,
                request_type="generate_until",
                text="step1 step2 #### 18",
            )
            for request in requests
        ]


def _model_artifact(sandbox: Sandbox) -> ModelArtifact:
    model_dir = sandbox.artifact_dir("model", "fake-planner-model")
    model_dir.mkdir(parents=True, exist_ok=True)
    return ModelArtifact(
        artifact_id="fake-planner-model",
        name="fake",
        path=model_dir,
        producer="test",
        meta={"base_model": "fake/qwen-stub"},
    )


def test_plan_then_execute_runs_gsm8k_via_keyword_intent(tmp_path):
    if not os.environ.get("HF_ENDPOINT") and not os.environ.get("HF_DATASETS_OFFLINE"):
        pytest.skip("HF_ENDPOINT not set; avoid network reach to huggingface.co")

    plan = plan_eval("评估模型数学能力", max_tasks=1)
    assert plan, "planner should produce at least one item"
    item = plan[0]
    assert item.task == "gsm8k"

    # Override defaults so the integration test stays small.
    light_plan = [
        EvalPlanItem(
            task=item.task,
            adapter=item.adapter,
            adapter_config=dict(item.adapter_config),
            limit=3,
            num_fewshot=0,
            batch_size=3,
            max_new_tokens=16,
            apply_chat_template=False,
            confirm_run_unsafe_code=item.confirm_run_unsafe_code,
            headline_metric=item.headline_metric,
            score=item.score,
            rationale=item.rationale,
        )
    ]

    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()
    model = _model_artifact(sandbox)
    report = execute_plan(
        light_plan,
        model_artifact=model,
        adapter_factory=lambda name: _FakeAdapter(),
    )

    assert report.model_id == model.artifact_id
    assert len(report.results) == 1
    result = report.results[0]
    assert result.ok, result.error
    assert result.headline_score is not None
    assert result.metrics, "expected at least one numeric metric"
    table = report.headline_table()
    assert table[0]["task"] == "gsm8k"
    assert "metric" in table[0]
