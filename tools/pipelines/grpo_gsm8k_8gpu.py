from __future__ import annotations

from dataos_train.spec import GRPO, Rollout
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

PIPELINE = PipelineDefinition(
    name="grpo_gsm8k_8gpu",
    backend="verl",
    dataset=DatasetPlan(
        kind="local_jsonl",
        source="DataOS-Operator/tests/fixtures/gsm8k_train_200.jsonl",
        sample_size=200,
        repeat_to=512,
    ),
    method_factory=constant_method(
        GRPO(
            train_batch=256,
            ppo_mini_batch=64,
            ppo_micro_batch_per_gpu=1,
            group_size=4,
            rollout=Rollout(
                n=4,
                max_prompt_len=512,
                max_response_len=512,
                gpu_memory_utilization=0.45,
                tensor_parallel_size=2,
            ),
            total_epochs=1,
            save_freq=1,
            test_freq=-1,
        )
    ),
    resources=Resources(nodes=1, gpus=8, cpus=32, time_h=6),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))
