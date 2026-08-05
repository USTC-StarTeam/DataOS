from __future__ import annotations

from dataos_train.spec import KTO
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

PIPELINE = PipelineDefinition(
    name="failure_kto_bad_feedback_data",
    backend="llamafactory",
    dataset=DatasetPlan(
        kind="inline_jsonl",
        records=({"prompt": "Say hi.", "response": "hi", "kto_tag": "maybe"},),
    ),
    method_factory=constant_method(
        KTO(
            finetuning_type="lora",
            template="qwen",
            lora_rank=8,
            lora_alpha=16,
            extra={"max_steps": 1, "logging_steps": 1},
        )
    ),
    resources=Resources(nodes=1, gpus=1, cpus=4, time_h=1),
    expected_failure=True,
    expected_failure_reason="KTO feedback tag must be boolean-like",
    acceptance_artifacts=(),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))

