"""verl GRPO 8-GPU stress run — sustained full-node load, 16 steps.

Doubles the rollout pressure of grpo_gsm8k_8gpu: group_size 8 (2048 sequences per
step at response_len 512), micro batch 2, vLLM at 0.5 GPU memory. 2048 rows /
train_batch 256 -> 8 steps/epoch, 2 epochs -> 16 optimizer steps (~1h on 8x3090).
Purpose: sustained utilization, memory headroom under load, and reward trend over
a longer horizon than the smoke runs.
"""

from __future__ import annotations

from dataos_train.spec import GRPO, Rollout
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

PIPELINE = PipelineDefinition(
    name="grpo_gsm8k_8gpu_stress",
    backend="verl",
    dataset=DatasetPlan(
        kind="local_jsonl",
        source="DataOS-Operator/tests/fixtures/gsm8k_train_200.jsonl",
        sample_size=200,
        repeat_to=2048,
    ),
    method_factory=constant_method(
        GRPO(
            train_batch=256,
            ppo_mini_batch=64,
            ppo_micro_batch_per_gpu=2,
            group_size=8,
            rollout=Rollout(
                n=8,
                max_prompt_len=512,
                max_response_len=512,
                gpu_memory_utilization=0.5,
                tensor_parallel_size=2,
            ),
            total_epochs=2,  # 2048/256 = 8 steps/epoch -> 16 steps
            save_freq=8,
            test_freq=-1,
        )
    ),
    resources=Resources(nodes=1, gpus=8, cpus=32, time_h=4),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))
