from __future__ import annotations

from dataos_train.spec import GRPO, Reward, Rollout
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

PIPELINE = PipelineDefinition(
    name="failure_grpo_bad_reward",
    backend="verl",
    dataset=DatasetPlan(
        kind="inline_jsonl",
        records=({"prompt": "What is 2+2?", "ground_truth": "4", "data_source": "bad"},),
    ),
    method_factory=constant_method(
        GRPO(
            train_batch=16,
            ppo_mini_batch=8,
            ppo_micro_batch_per_gpu=1,
            group_size=2,
            rollout=Rollout(n=2, tensor_parallel_size=1),
            reward=Reward(ref="bad_reward"),
        )
    ),
    resources=Resources(nodes=1, gpus=1, cpus=4, time_h=1),
    expected_failure=True,
    expected_failure_reason="unknown reward.ref must fail dataos-train validation",
    acceptance_artifacts=(),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))

