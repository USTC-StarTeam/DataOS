"""Render the LLaMA-Factory training config.

Emitted as YAML: LF 0.9.x's ``read_args`` only loads ``.yaml``/``.yml`` correctly
(``OmegaConf.load(Path)``); its ``.json`` branch is broken (``json.load(Path)`` raises
``AttributeError``). ``to_yaml`` formats floats as plain decimals so scientific notation
(e.g. ``1e-6``) is never parsed as a string.
"""

from __future__ import annotations

import json
from typing import Any

from ...spec import DPO, KTO, ORPO, PPO, SFT, TrainSpec

LlamaFactoryMethod = SFT | DPO | PPO | KTO | ORPO

SUPPORTED_METHODS = ("sft", "dpo", "ppo", "kto", "orpo")


def method_stage(method: LlamaFactoryMethod) -> str:
    """Return the native LLaMA-Factory stage for a DataOS method."""
    return "dpo" if method.kind == "orpo" else method.kind


def method_dataset_name(method: LlamaFactoryMethod) -> str:
    return f"dataos_{method.kind}"


def method_data_subdir(method: LlamaFactoryMethod) -> str:
    return method.kind


def render_train_config(
    spec: TrainSpec,
    method: LlamaFactoryMethod,
    data_dir: str,
    output_dir: str,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    dataset_name = dataset_name or method_dataset_name(method)
    cfg: dict[str, Any] = {
        "model_name_or_path": spec.base_model,  # the injected parent; never hardcoded
        "stage": method_stage(method),
        "do_train": True,
        "finetuning_type": method.finetuning_type,
        "dataset": dataset_name,
        "dataset_dir": data_dir,
        "template": method.template,
        "cutoff_len": method.cutoff_len,
        "learning_rate": method.learning_rate,
        "num_train_epochs": method.num_train_epochs,
        "per_device_train_batch_size": method.per_device_train_batch_size,
        "gradient_accumulation_steps": method.gradient_accumulation_steps,
        "warmup_ratio": method.warmup_ratio,
        "bf16": method.bf16,
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "logging_steps": 10,
        "save_strategy": "no",  # final model saved at end; skip intermediate checkpoints
        "report_to": "none",
    }
    if getattr(method, "save_steps", None):
        # Sample the training trajectory. `save_only_model` is forced, not offered: the optimizer
        # state is ~4x the weights (fp32 master + Adam m,v) and we never resume a run — a
        # checkpoint here exists to be EVALUATED and to be continued from as a fresh run, both of
        # which need weights alone. With it, a checkpoint costs one model's worth of disk.
        cfg.update(
            {
                "save_strategy": "steps",
                "save_steps": method.save_steps,
                "save_total_limit": method.save_total_limit,
                "save_only_model": True,
            }
        )
    if method.finetuning_type == "lora":
        cfg["lora_target"] = method.lora_target
        cfg["lora_rank"] = method.lora_rank
        cfg["lora_alpha"] = method.lora_alpha
    if method.finetuning_type == "freeze" and method.freeze_trainable_layers is not None:
        cfg["freeze_trainable_layers"] = method.freeze_trainable_layers
    if method.quantization_bit is not None:
        cfg["quantization_bit"] = method.quantization_bit  # QLoRA (bitsandbytes 4-bit)

    _add_method_config(cfg, method)
    cfg.update(method.extra)  # any extra LF training key the plan asked for
    if isinstance(spec.overrides, dict):
        cfg.update(spec.overrides)  # raw escape hatch, last-wins
    # Re-assert DataOS-owned LF keys after escape hatches so render remains deterministic.
    cfg.update(
        {
            "model_name_or_path": spec.base_model,
            "stage": method_stage(method),
            "finetuning_type": method.finetuning_type,
            "dataset": dataset_name,
            "dataset_dir": data_dir,
            "template": method.template,
            "output_dir": output_dir,
        }
    )
    return cfg


def _add_method_config(cfg: dict[str, Any], method: LlamaFactoryMethod) -> None:
    if isinstance(method, DPO):
        cfg["pref_loss"] = "sigmoid"
        cfg["pref_beta"] = method.pref_beta
        cfg["pref_ftx"] = method.pref_ftx
        cfg["dpo_label_smoothing"] = method.dpo_label_smoothing
        _add_reference_model(cfg, method)
    elif isinstance(method, ORPO):
        cfg["pref_loss"] = "orpo"
        cfg["pref_beta"] = method.pref_beta
    elif isinstance(method, KTO):
        cfg["pref_beta"] = method.pref_beta
        cfg["kto_chosen_weight"] = method.kto_chosen_weight
        cfg["kto_rejected_weight"] = method.kto_rejected_weight
        _add_reference_model(cfg, method)
    elif isinstance(method, PPO):
        cfg["reward_model"] = method.reward_model
        cfg["reward_model_type"] = method.reward_model_type
        cfg["reward_model_adapters"] = method.reward_model_adapters
        cfg["reward_model_quantization_bit"] = method.reward_model_quantization_bit
        cfg["ppo_buffer_size"] = method.ppo_buffer_size
        cfg["ppo_epochs"] = method.ppo_epochs
        cfg["ppo_score_norm"] = method.ppo_score_norm
        cfg["ppo_target"] = method.ppo_target
        cfg["ppo_whiten_rewards"] = method.ppo_whiten_rewards
        cfg["max_new_tokens"] = method.max_new_tokens
        cfg["temperature"] = method.temperature
        cfg["top_p"] = method.top_p
        _add_reference_model(cfg, method)

    for key in list(cfg):
        if cfg[key] is None:
            cfg.pop(key)


def _add_reference_model(cfg: dict[str, Any], method: DPO | PPO | KTO) -> None:
    cfg["ref_model"] = method.ref_model
    cfg["ref_model_adapters"] = method.ref_model_adapters
    cfg["ref_model_quantization_bit"] = method.ref_model_quantization_bit


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float):
        s = repr(v)
        if "e" in s or "E" in s:  # expand sci-notation so YAML reads a float, not a string
            s = f"{v:.12f}".rstrip("0").rstrip(".")
        return s
    if isinstance(v, int):
        return str(v)
    s = str(v)
    unsafe = s == "" or s[0] in "!&*?|>%@`\"'#,[]{}" or ": " in s or s.strip() != s
    if unsafe or s.lower() in ("true", "false", "null", "yes", "no", "~"):
        return json.dumps(s, ensure_ascii=False)
    return s


def to_yaml(cfg: dict[str, Any]) -> str:
    """Minimal YAML for a flat scalar/list config (what LF training configs are)."""
    lines: list[str] = []
    for key, val in cfg.items():
        if isinstance(val, list):
            lines.append(f"{key}:")
            lines.extend(f"  - {_yaml_scalar(it)}" for it in val)
        else:
            lines.append(f"{key}: {_yaml_scalar(val)}")
    return "\n".join(lines) + "\n"
