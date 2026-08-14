"""The agent-facing ``TrainSpec`` and its method union.

An agent emits a ``TrainSpec`` (facts only: backend, model, data, method, resources) and
gets back a ``TrainResult``. Field names reuse the existing DataOS ``ExperimentPlan.train``
keys so a spec lines up with today's plan/report shapes.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from dataos_launch.model import Model
from dataos_launch.spec import Spec
from pydantic import Field


class DatasetRef(Model):
    """Pointer to already-built canonical data on disk (offline operators run upstream)."""

    path: str
    format: Literal["auto", "alpaca", "messages", "sharegpt", "rl", "canonical"] = "auto"


class EnvSpec(Model):
    """Conda / PYTHONPATH activation.

    ``backends`` maps an **env domain** to its conda env (name or prefix path). A domain is
    usually just the backend name (``"llamafactory"``, ``"verl"``); a method whose dependency
    set conflicts with the backend default gets a method-scoped domain such as
    ``"llamafactory:ppo"``, and a model family whose architecture needs a different
    transformers gets a family-scoped one such as ``"llamafactory@smollm3"``. Both take
    precedence over the plain backend domain; see :meth:`env_for` for the order. Backends
    advertise their non-default method domains in ``Capabilities.env_domains``; family domains
    come from the environment config alone, because the set of trainable families is a
    property of the machine's envs, not of the backend.
    """

    conda_sh: str | None = None
    backends: dict[str, str] = Field(
        default_factory=dict
    )  # env domain -> conda env, e.g. {"llamafactory": "Llama", "llamafactory:ppo": "LlamaPPO"}
    deploy_root: str | None = None  # $AGENTVER; pythonpath defaults to {root}/src:{root}
    pythonpath: list[str] = Field(default_factory=list)
    extra: dict[str, str] = Field(default_factory=dict)

    def env_for(
        self,
        backend: str,
        method_kind: str | None = None,
        model_family: str | None = None,
    ) -> str | None:
        """Resolve the env for a (backend, method, model family); most specific wins.

        ``model_family`` is the model's HF ``model_type``. It selects an env because
        architecture support is a *dependency* fact: SmolLM3 needs transformers >= 4.53, so an
        env pinned below that cannot train it no matter what the spec says. Declaring that in
        the environment config keeps the tool free of a model table — it only reads
        ``model_type`` off the model itself and asks this method.
        """
        keys = [
            f"{backend}:{method_kind}@{model_family}" if method_kind and model_family else "",
            f"{backend}:{method_kind}" if method_kind else "",
            f"{backend}@{model_family}" if model_family else "",
            backend,
        ]
        for key in keys:
            if key and self.backends.get(key):
                return self.backends[key]
        return None


class OnlineOpSpec(Model):
    """A data operator that interacts with the training loop (online).

    Realized by the backend at the given ``hook`` point (offline operators run upstream
    and are not represented here). ``hook=None`` selects the op's own default hook point;
    the protocol is defined in :mod:`dataos_train.online` (see docs/online_ops.md).
    """

    op: str
    params: dict[str, Any] = Field(default_factory=dict)
    hook: Literal["select", "reweight", "step"] | None = None


class SFT(Model):
    """SFT hyperparameters; forwarded ~1:1 to LLaMA-Factory YAML. Omit model_name_or_path.

    finetuning_type: ``lora`` (adapter, merged after), ``full`` (all params), ``freeze``
    (train only the top ``freeze_trainable_layers`` blocks). Set ``quantization_bit=4`` with
    ``finetuning_type=lora`` for QLoRA.
    """

    kind: Literal["sft"] = "sft"
    finetuning_type: Literal["lora", "full", "freeze"] = "lora"
    # None = derive from the model's config.json model_type (ext/templates.py).
    # There is deliberately no default family: a wrong chat template is a silent
    # failure — capped scores, nothing in any log.
    template: str | None = None
    learning_rate: float = 1e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    cutoff_len: int = 1024
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target: str = "all"
    quantization_bit: int | None = None  # 4 -> QLoRA (needs bitsandbytes)
    freeze_trainable_layers: int | None = None  # freeze-tuning: # of top blocks to train
    warmup_ratio: float = 0.0
    bf16: bool = True
    #: Save a checkpoint every N optimizer steps. None (default) keeps ``save_strategy: "no"`` —
    #: only the final model. Set it to sample the training TRAJECTORY: which point along one run
    #: is best is a question no amount of data-recipe search can answer, and the hand-driven
    #: gsm8k reference beat us by 11.2 points partly on "the midpoint at step 459 outperformed
    #: the final checkpoint". Each retained checkpoint costs one model's worth of disk (weights
    #: only — see save_only_model in the LF config), which ``cli.precheck_disk`` accounts for.
    save_steps: int | None = None
    #: How many checkpoints to keep (oldest deleted first). Ignored unless save_steps is set.
    save_total_limit: int = 3
    extra: dict[str, Any] = Field(default_factory=dict)  # any extra LF training key, forwarded


class _LlamaFactoryTuning(Model):
    """Common LLaMA-Factory tuning controls DataOS-train commits to exposing."""

    finetuning_type: Literal["lora", "full", "freeze"] = "lora"
    # None = derive from the model's config.json model_type (ext/templates.py).
    # There is deliberately no default family: a wrong chat template is a silent
    # failure — capped scores, nothing in any log.
    template: str | None = None
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    cutoff_len: int = 1024
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target: str = "all"
    quantization_bit: int | None = None
    freeze_trainable_layers: int | None = None
    warmup_ratio: float = 0.0
    bf16: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)


class _ReferenceModelMixin(Model):
    """Optional reference model controls used by DPO/KTO/PPO."""

    ref_model: str | None = None
    ref_model_adapters: str | None = None
    ref_model_quantization_bit: Literal[4, 8] | None = None


class DPO(_LlamaFactoryTuning, _ReferenceModelMixin):
    """Direct Preference Optimization over prompt/chosen/rejected preference data."""

    kind: Literal["dpo"] = "dpo"
    pref_beta: float = 0.1
    pref_ftx: float = 0.0
    dpo_label_smoothing: float = 0.0


class ORPO(_LlamaFactoryTuning):
    """Odds Ratio Preference Optimization over prompt/chosen/rejected preference data."""

    kind: Literal["orpo"] = "orpo"
    pref_beta: float = 0.1


class KTO(_LlamaFactoryTuning, _ReferenceModelMixin):
    """Kahneman-Tversky Optimization over response data with boolean ``kto_tag``."""

    kind: Literal["kto"] = "kto"
    pref_beta: float = 0.1
    kto_chosen_weight: float = 1.0
    kto_rejected_weight: float = 1.0


class PPO(_LlamaFactoryTuning, _ReferenceModelMixin):
    """PPO over prompt data, scored by a LLaMA-Factory reward model."""

    kind: Literal["ppo"] = "ppo"
    reward_model: str | None = None
    reward_model_type: Literal["lora", "full", "api"] = "lora"
    reward_model_adapters: str | None = None
    reward_model_quantization_bit: Literal[4, 8] | None = None
    ppo_buffer_size: int = 1
    ppo_epochs: int = 4
    ppo_score_norm: bool = False
    ppo_target: float = 6.0
    ppo_whiten_rewards: bool = False
    max_new_tokens: int = 1024
    temperature: float = 0.95
    top_p: float = 0.7


class Rollout(Model):
    engine: Literal["vllm"] = "vllm"
    n: int = 5
    max_prompt_len: int = 1024
    max_response_len: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    gpu_memory_utilization: float = 0.4
    tensor_parallel_size: int = 1


class Reward(Model):
    """RL reward. ``path`` is RUNTIME-OWNED: an absolute path to a reward file defining
    ``compute_score(data_source, solution_str, ground_truth, ...) -> float`` — typically
    WRITTEN BY THE DRIVING AGENT from the task's frozen format contract (verl loads it by
    path; this library ships no task rewards). ``ref`` is just the bench/data_source label."""

    kind: Literal["verifier"] = "verifier"
    ref: str = "gsm8k"  # bench / data_source label (gt extraction + row tagging)
    path: str | None = None  # runtime-owned reward file path (required by the verl backend)


class GRPO(Model):
    """GRPO/RL hyperparameters; map onto verl's ppo_trainer config."""

    kind: Literal["grpo"] = "grpo"
    learning_rate: float = 1e-6
    kl_coef: float = 1e-3
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0
    grad_clip: float = 1.0
    train_batch: int = 256
    ppo_mini_batch: int = 64
    ppo_micro_batch_per_gpu: int = 4
    group_size: int = 5  # validate() enforces == rollout.n
    adv_estimator: Literal["grpo"] = "grpo"
    finetuning_type: Literal["full", "lora"] = "full"
    lora_rank: int = 0
    lora_alpha: int = 16
    total_epochs: int = 1
    save_freq: int = -1
    test_freq: int = -1
    rollout: Rollout = Field(default_factory=Rollout)
    reward: Reward = Field(default_factory=Reward)
    extra: dict[str, Any] = Field(default_factory=dict)


class VerlPPO(Model):
    """Classic PPO (GAE + learned critic) on verl's ppo_trainer.

    Unlike GRPO there is no group baseline: ``rollout.n`` defaults to 1 and a critic model
    (defaults to ``base_model``) estimates values. The KL penalty moves into the reward
    (``use_kl_in_reward``), the canonical RLHF-PPO setup. The reward is the same verifier
    registry GRPO uses, so reward == eval holds here too.
    """

    kind: Literal["verl_ppo"] = "verl_ppo"
    learning_rate: float = 1e-6
    critic_learning_rate: float = 1e-5
    critic_model: str | None = None  # defaults to base_model
    critic_warmup: int = 0
    kl_coef: float = 1e-3  # algorithm.kl_ctrl.kl_coef (KL penalty applied in the reward)
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl"] = "kl"
    gamma: float = 1.0
    lam: float = 1.0
    entropy_coef: float = 0.0
    grad_clip: float = 1.0
    train_batch: int = 256
    ppo_mini_batch: int = 64
    ppo_micro_batch_per_gpu: int = 4
    adv_estimator: Literal["gae"] = "gae"
    finetuning_type: Literal["full"] = "full"
    total_epochs: int = 1
    save_freq: int = -1
    test_freq: int = -1
    rollout: Rollout = Field(default_factory=lambda: Rollout(n=1))
    reward: Reward = Field(default_factory=Reward)
    extra: dict[str, Any] = Field(default_factory=dict)


Method = Annotated[SFT | DPO | PPO | KTO | ORPO | GRPO | VerlPPO, Field(discriminator="kind")]


class TrainSpec(Spec):
    """The one spec an agent writes by hand — the method is what DataOS searches.

    ``name``/``run_dir``/``resources`` come from :class:`~dataos_launch.spec.Spec`; what is
    special here is ``method``, and nothing else. It also carried a ``cluster`` block until
    2026-08-14 — partition, QOS, node, filled by the brain from ``env.yaml`` and forwarded to the
    launcher on the Job. That was a third hand-written copy of a rule the other two instruments
    read from config, and it went stale the way a third copy does: when ``qos`` moved into the
    run config, this one kept reading a field that no longer existed and its jobs went to the
    account default in silence. Placement is now the launcher's, resolved at submit from
    ``--repo``/``--environment`` plus the run's own choices.
    """

    stage = "train"

    backend: str = "llamafactory"
    base_model: str  # resolved HF path, injected by DataOS (never written by the LLM)
    dataset: DatasetRef | None = None
    method: Method = Field(default_factory=SFT)
    online_ops: list[OnlineOpSpec] = Field(default_factory=list)
    env: EnvSpec = Field(default_factory=EnvSpec)
    overrides: Any = Field(default_factory=dict)  # dict|list: raw native-config escape hatch
