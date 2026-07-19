"""Build the dataos-train TrainSpec JSON from the resolved runtime contract + plan.

Pure dict construction — the brain does NOT import dataos_train (zero build-time
dependency; the spec crosses the boundary as data, per runtime-contract.md §12).
Runtime-owned fields (base_model / run_dir / dataset.path / env / cluster) are
injected HERE from the resolved contract — single source of truth. The planner
only ever contributes the ``method`` body (plan.train); unknown train keys are
forwarded via ``method.extra`` (dataos-train validates them).
"""

from __future__ import annotations

# SFT fields dataos-train's method schema knows natively; everything else -> extra.
_SFT_FIELDS = {
    "finetuning_type", "template", "learning_rate", "num_train_epochs",
    "per_device_train_batch_size", "gradient_accumulation_steps", "cutoff_len",
    "lora_rank", "lora_alpha", "lora_target", "quantization_bit",
    "freeze_trainable_layers", "warmup_ratio", "bf16",
}
# runtime-owned / redundant keys the planner must not smuggle into the method
_STRIP = {"model_name_or_path", "stage", "do_train", "dataset", "dataset_dir",
          "output_dir", "overwrite_output_dir"}


def build_train_spec(resolved: dict, *, name: str, plan_train: dict, parent_model: str,
                     run_root: str, time_h: float) -> dict:
    """TrainSpec dict for dataos-train (SFT via llamafactory)."""
    env = resolved.get("environment", {})
    run = resolved.get("run", {})
    tools = env.get("tools", {})
    train_env = tools.get("train_env", {})
    train_env_path = train_env.get("path") if isinstance(train_env, dict) else train_env
    resource = run.get("resource", {}) or {}
    gpus = int(resource.get("gpus") or 1)
    cpus = int(resource.get("cpus") or 8)

    method: dict = {"kind": "sft"}
    extra: dict = {}
    for key, val in (plan_train or {}).items():
        if key in _STRIP:
            continue
        (method if key in _SFT_FIELDS else extra)[key] = val
    if extra:
        method["extra"] = extra

    return {
        "name": name,
        "backend": "llamafactory",
        "base_model": parent_model,
        "run_dir": run_root,
        "dataset": {"path": f"{run_root}/data/sft", "format": "alpaca"},
        "method": method,
        "resources": {"nodes": 1, "gpus": gpus, "cpus": cpus, "time_h": time_h},
        "env": {
            "conda_sh": tools.get("conda_sh"),
            "backends": {"llamafactory": train_env_path},
        },
        "cluster": {
            "partition": env.get("slurm", {}).get("partition"),
            "qos": env.get("slurm", {}).get("qos"),
            "nodelist": resource.get("node"),
            "gres": f"gpu:{gpus}",
            "cpus_per_task": cpus,
        },
    }
