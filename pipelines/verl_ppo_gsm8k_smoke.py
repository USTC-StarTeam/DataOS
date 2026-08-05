"""verl classic PPO (GAE + critic) on GSM8K — multi-step functional validation.

128 prompts, train_batch 32, 2 epochs -> 8 optimizer steps on 2 GPUs: enough to observe
the critic warm up (vf_loss moving) and the KL-in-reward penalty behaving, not just the
plumbing. Critic defaults to the base model; reward is the same gsm8k verifier eval uses.
"""

from __future__ import annotations

from dataos_train.spec import Rollout, VerlPPO
from dataos_train.types import Resources
from pipelines.common import DatasetPlan, PipelineDefinition, constant_method, run_pipeline

PIPELINE = PipelineDefinition(
    name="verl_ppo_gsm8k_smoke",
    backend="verl",
    dataset=DatasetPlan(
        kind="local_jsonl",
        source="DataOS-Operator/tests/fixtures/gsm8k_train_200.jsonl",
        sample_size=128,
    ),
    method_factory=constant_method(
        VerlPPO(
            train_batch=32,
            ppo_mini_batch=16,
            ppo_micro_batch_per_gpu=1,
            critic_warmup=0,
            rollout=Rollout(
                n=1,
                max_prompt_len=512,
                max_response_len=256,
                gpu_memory_utilization=0.35,  # critic shares the GPUs; leave FSDP headroom
                tensor_parallel_size=1,
            ),
            total_epochs=2,  # 128/32 = 4 steps/epoch -> 8 steps total
            save_freq=4,
            test_freq=-1,
        )
    ),
    resources=Resources(nodes=1, gpus=2, cpus=16, time_h=3),
)


if __name__ == "__main__":
    raise SystemExit(run_pipeline(PIPELINE))
