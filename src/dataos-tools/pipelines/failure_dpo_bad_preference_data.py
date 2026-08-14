from __future__ import annotations

from dataos_train.spec import DPO
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

PIPELINE = PipelineDefinition(
    name="failure_dpo_bad_preference_data",
    backend="llamafactory",
    dataset=DatasetPlan(
        kind="inline_jsonl",
        records=({"prompt": "Which answer is better?", "chosen": "Only one side is present."},),
    ),
    method_factory=constant_method(
        DPO(
            finetuning_type="lora",
            template="qwen",
            lora_rank=8,
            lora_alpha=16,
            extra={"max_steps": 1, "logging_steps": 1},
        )
    ),
    resources=Resources(nodes=1, gpus=1, cpus=4, time_h=1),
    expected_failure=True,
    expected_failure_reason="preference data missing rejected response must fail conversion",
    acceptance_artifacts=(),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))

