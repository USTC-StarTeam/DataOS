from __future__ import annotations

from dataos_train.spec import SFT
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

PIPELINE = PipelineDefinition(
    name="sft_full_micro",
    backend="llamafactory",
    dataset=DatasetPlan(
        kind="local_jsonl",
        source="DataOS-Operator/tests/fixtures/alpaca_200.jsonl",
        sample_size=32,
    ),
    method_factory=constant_method(
        SFT(
            finetuning_type="full",
            template="qwen",
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            cutoff_len=512,
            extra={"max_steps": 1, "logging_steps": 1},
        )
    ),
    resources=Resources(nodes=1, gpus=1, cpus=8, time_h=1),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))

