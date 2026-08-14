from __future__ import annotations

from dataos_train.spec import SFT
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

PIPELINE = PipelineDefinition(
    name="sft_qlora_alpaca_smoke",
    backend="llamafactory",
    dataset=DatasetPlan(
        kind="local_jsonl",
        source="DataOS-Operator/tests/fixtures/alpaca_200.jsonl",
        sample_size=64,
    ),
    method_factory=constant_method(
        SFT(
            finetuning_type="lora",
            template="qwen",
            learning_rate=1e-4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            cutoff_len=1024,
            lora_rank=16,
            lora_alpha=32,
            quantization_bit=4,
            extra={"max_steps": 5, "logging_steps": 1},
        )
    ),
    resources=Resources(nodes=1, gpus=1, cpus=8, time_h=1),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))

