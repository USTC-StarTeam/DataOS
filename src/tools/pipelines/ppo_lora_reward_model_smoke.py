from __future__ import annotations

from dataos_train.spec import PPO
from dataos_train.types import Resources
from pipelines.common import (
    DEFAULT_LF_PPO_ENV,
    DatasetPlan,
    PipelineContext,
    PipelineDefinition,
    run_pipeline,
)


def method(ctx: PipelineContext) -> PPO:
    return PPO(
        finetuning_type="lora",
        template="qwen",
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        cutoff_len=1024,
        lora_rank=16,
        lora_alpha=32,
        reward_model=ctx.require_reward_model(),
        reward_model_type="full",
        ppo_buffer_size=1,
        ppo_epochs=1,
        max_new_tokens=256,
        extra={"max_steps": 2, "logging_steps": 1},
    )


PIPELINE = PipelineDefinition(
    name="ppo_lora_reward_model_smoke",
    backend="llamafactory",
    dataset=DatasetPlan(
        kind="hf_ppo",
        source="HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
        sample_size=128,
    ),
    method_factory=method,
    resources=Resources(nodes=1, gpus=1, cpus=8, time_h=1),
    backend_env=DEFAULT_LF_PPO_ENV,
    env_extra={
        # LLaMA-Factory 0.9.5 has conflicting global and PPO-local TRL version checks.
        "DISABLE_VERSION_CHECK": "1",
    },
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))
