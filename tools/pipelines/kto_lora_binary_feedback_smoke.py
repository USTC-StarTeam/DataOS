from __future__ import annotations

from dataos_train.spec import KTO
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

PIPELINE = PipelineDefinition(
    name="kto_lora_binary_feedback_smoke",
    backend="llamafactory",
    dataset=DatasetPlan(
        kind="hf_kto",
        source="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        sample_size=512,
    ),
    method_factory=constant_method(
        KTO(
            finetuning_type="lora",
            template="qwen",
            learning_rate=5e-6,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            cutoff_len=1024,
            lora_rank=16,
            lora_alpha=32,
            pref_beta=0.1,
            extra={"max_steps": 5, "logging_steps": 1},
        )
    ),
    resources=Resources(nodes=1, gpus=1, cpus=8, time_h=1),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))

