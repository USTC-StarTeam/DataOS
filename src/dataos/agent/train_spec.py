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


def _slurm_time_hours(v: str) -> float:
    """Slurm time -> hours: [[days-]hours:]minutes[:seconds]; a bare number is minutes."""
    days, _, rest = v.partition("-")
    if not rest:
        days, rest = "0", v
    if ":" not in rest:
        return int(days) * 24 + int(rest) / 60
    secs = 0
    for part in rest.split(":"):
        secs = secs * 60 + int(part)
    return (int(days) * 86400 + secs) / 3600


def build_train_spec(resolved: dict, *, name: str, plan_train: dict, parent_model: str,
                     run_root: str, time_h: float) -> dict:
    """TrainSpec dict for dataos-train. ``env.backends`` carries the FULL env menu derived
    from the environment's ``tools.*.provides`` (the single source); dataos-train's
    ``env_for()`` resolves method→env at build. The backend itself is the code default
    until the plan can express a method choice. ``time_h`` is clamped to the partition's
    slurm.max_time — a longer request pends forever on PartitionTimeLimit."""
    env = resolved.get("environment", {})
    run = resolved.get("run", {})
    tools = env.get("tools", {})
    max_time = (env.get("slurm") or {}).get("max_time")
    if max_time:
        time_h = min(time_h, _slurm_time_hours(max_time))
    backend = "llamafactory"  # code default (plan-driven backend selection: verl work)
    env_backends = {
        b: (tools.get(name, {}).get("path") if isinstance(tools.get(name), dict) else tools.get(name))
        for name in tools
        for b in ("llamafactory", "verl")
        if b in ((tools.get(name) or {}).get("provides") or [])
    }
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
        "backend": backend,
        "base_model": parent_model,
        "run_dir": run_root,
        "dataset": {"path": f"{run_root}/data/sft", "format": "alpaca"},
        "method": method,
        "resources": {"nodes": 1, "gpus": gpus, "cpus": cpus, "time_h": time_h},
        "env": {
            "backends": env_backends,
        },
        "cluster": {
            "partition": env.get("slurm", {}).get("partition"),
            "qos": env.get("slurm", {}).get("qos"),
            # run pin wins; else the env's node pool (env.slurm.nodes)
            "nodelist": resource.get("node") or env.get("slurm", {}).get("nodes"),
            "gres": f"gpu:{gpus}",
            "cpus_per_task": cpus,
        },
    }
