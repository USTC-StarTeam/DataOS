from __future__ import annotations

from dataos_train.spec import PPO
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

PIPELINE = PipelineDefinition(
    name="failure_ppo_missing_reward_dependency",
    backend="llamafactory",
    dataset=DatasetPlan(kind="inline_jsonl", records=({"prompt": "What is 2+2?"},)),
    method_factory=constant_method(
        PPO(
            finetuning_type="lora",
            template="qwen",
            reward_model=None,
            lora_rank=8,
            lora_alpha=16,
        )
    ),
    resources=Resources(nodes=1, gpus=1, cpus=4, time_h=1),
    expected_failure=True,
    expected_failure_reason="PPO must fail validation when reward_model is absent",
    acceptance_artifacts=(),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))

