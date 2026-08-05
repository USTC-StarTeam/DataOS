from __future__ import annotations

from dataos_train.spec import GRPO, Reward, Rollout
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

AIME_RECORDS = (
    {"prompt": "Find the value of 7^2 + 24.", "ground_truth": "73", "data_source": "aime"},
    {"prompt": "If x + 5 = 17, what is x?", "ground_truth": "12", "data_source": "aime"},
    {"prompt": "Compute 13 * 13.", "ground_truth": "169", "data_source": "aime"},
    {
        "prompt": "What is the remainder when 100 is divided by 9?",
        "ground_truth": "1",
        "data_source": "aime",
    },
)


PIPELINE = PipelineDefinition(
    name="grpo_aime_reward_smoke",
    backend="verl",
    dataset=DatasetPlan(kind="inline_jsonl", records=AIME_RECORDS, sample_size=4, repeat_to=32),
    method_factory=constant_method(
        GRPO(
            train_batch=32,
            ppo_mini_batch=16,
            ppo_micro_batch_per_gpu=1,
            group_size=4,
            rollout=Rollout(
                n=4,
                max_prompt_len=512,
                max_response_len=256,
                gpu_memory_utilization=0.45,
                tensor_parallel_size=1,
            ),
            reward=Reward(ref="aime"),
            total_epochs=1,
            save_freq=1,
            test_freq=-1,
        )
    ),
    resources=Resources(nodes=1, gpus=1, cpus=12, time_h=2),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))
