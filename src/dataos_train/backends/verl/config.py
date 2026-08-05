"""Render verl ``main_ppo`` config as dotted CLI overrides (no vendored base yaml).

We pass deltas over verl's packaged base config; ``spec.overrides`` is appended verbatim
(last-wins), so any verl knob is reachable. This dotted-override list is the brittle,
version-sensitive surface — isolated here so a verl bump touches one file.
Verified against verl 0.4.1 (ppo_trainer.yaml).
"""

from __future__ import annotations

from ...spec import GRPO, TrainSpec, VerlPPO
from ...types import Resources


def _b(x: bool) -> str:
    return "true" if x else "false"


def render_overrides(
    spec: TrainSpec,
    method: GRPO | VerlPPO,
    base_model: str,
    resources: Resources,
    reward_path: str,
    data_dir: str,
) -> list[str]:
    m = method
    ov = [
        # data
        f"data.train_files={data_dir}/train.parquet",
        f"data.val_files={data_dir}/val.parquet",
        f"data.train_batch_size={m.train_batch}",
        f"data.max_prompt_length={m.rollout.max_prompt_len}",
        f"data.max_response_length={m.rollout.max_response_len}",
        f"algorithm.adv_estimator={m.adv_estimator}",
        # actor / policy
        f"actor_rollout_ref.model.path={base_model}",
        f"actor_rollout_ref.actor.optim.lr={m.learning_rate}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={m.ppo_mini_batch}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={m.ppo_micro_batch_per_gpu}",
        f"actor_rollout_ref.actor.entropy_coeff={m.entropy_coef}",
        f"actor_rollout_ref.actor.grad_clip={m.grad_clip}",
        # rollout (vLLM)
        f"actor_rollout_ref.rollout.name={m.rollout.engine}",
        f"actor_rollout_ref.rollout.n={m.rollout.n}",
        f"actor_rollout_ref.rollout.temperature={m.rollout.temperature}",
        f"actor_rollout_ref.rollout.top_p={m.rollout.top_p}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={m.rollout.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={m.rollout.tensor_parallel_size}",
        # log-prob recompute micro-batches (verl requires one of total/per_gpu for ref + rollout)
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={m.ppo_micro_batch_per_gpu}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={m.ppo_micro_batch_per_gpu}",
        # reward (reuses the eval verifiers)
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        # trainer / placement (console logger: compute nodes are offline)
        "trainer.logger=[console]",
        "trainer.project_name=dataos_train",
        f"trainer.experiment_name={spec.name}",
        f"trainer.n_gpus_per_node={resources.gpus}",
        f"trainer.nnodes={resources.nodes}",
        f"trainer.total_epochs={m.total_epochs}",
        f"trainer.save_freq={m.save_freq}",
        f"trainer.test_freq={m.test_freq}",
        f"trainer.default_local_dir={spec.run_dir}/verl/ckpt",
        "trainer.val_before_train=false",
    ]
    if isinstance(m, GRPO):
        # GRPO: KL in the loss, not the reward; group baseline instead of a critic
        ov += [
            "algorithm.use_kl_in_reward=false",
            "actor_rollout_ref.actor.use_kl_loss=true",
            f"actor_rollout_ref.actor.kl_loss_coef={m.kl_coef}",
            f"actor_rollout_ref.actor.kl_loss_type={m.kl_loss_type}",
        ]
        if m.finetuning_type == "lora":
            ov += [
                f"actor_rollout_ref.model.lora_rank={m.lora_rank}",
                f"actor_rollout_ref.model.lora_alpha={m.lora_alpha}",
            ]
    else:
        # PPO/GAE: KL penalty in the reward, learned critic estimates values
        ov += [
            "algorithm.use_kl_in_reward=true",
            f"algorithm.kl_ctrl.kl_coef={m.kl_coef}",
            f"algorithm.kl_penalty={m.kl_penalty}",
            f"algorithm.gamma={m.gamma}",
            f"algorithm.lam={m.lam}",
            "actor_rollout_ref.actor.use_kl_loss=false",
            f"critic.model.path={m.critic_model or base_model}",
            f"critic.optim.lr={m.critic_learning_rate}",
            f"critic.ppo_micro_batch_size_per_gpu={m.ppo_micro_batch_per_gpu}",
            f"trainer.critic_warmup={m.critic_warmup}",
        ]
    if isinstance(spec.overrides, list):
        ov += [str(x) for x in spec.overrides]
    elif isinstance(spec.overrides, dict):
        ov += [f"{k}={v}" for k, v in spec.overrides.items()]
    return ov
