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
#: Plan keys that are TYPED fields of dataos-train's SFT method; anything else is forwarded
#: through ``method.extra``. The distinction is not cosmetic — dataos-train reasons about the
#: typed fields (``cli.precheck_disk`` sizes the run from ``save_steps``/``save_total_limit``,
#: for one), and a key that lands in ``extra`` still reaches LLaMA-Factory but is invisible to
#: that reasoning. This set mirrors ``dataos_train.spec.SFT``; ``tests/agent/
#: test_train_spec_contract.py`` fails if the two drift apart.
_SFT_FIELDS = {
    "finetuning_type", "learning_rate", "num_train_epochs",
    "per_device_train_batch_size", "gradient_accumulation_steps", "cutoff_len",
    "lora_rank", "lora_alpha", "lora_target", "quantization_bit",
    "freeze_trainable_layers", "warmup_ratio", "bf16",
    "save_steps", "save_total_limit",
}
# runtime-owned / redundant keys the planner must not smuggle into the method.
# `template` is here because the chat template is NOT a knob: it is a function of the model, and
# the model comes from the run config. A plan that could name it could break train/eval parity,
# and one did — the 2026-08-11 gsm8k batch's top score was produced under another family's
# template (see docs/audit-2026-08-08.md 13.2). dataos-train derives it from the model's own
# model_type; nothing on this side may express it.
_STRIP = {"model_name_or_path", "stage", "do_train", "dataset", "dataset_dir",
          "output_dir", "overwrite_output_dir", "template"}


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
    env_backends: dict[str, str] = {}
    for name in tools:
        tool = tools.get(name)
        tool = tool if isinstance(tool, dict) else {"path": tool}
        for b in ("llamafactory", "verl"):
            if b not in (tool.get("provides") or []):
                continue
            # An env that lists `model_families` claims ONLY those families' domains — the
            # architectures it can actually train are a property of its transformers pin, and
            # falling back to the unscoped domain for an undeclared family is how a job dies
            # at model load instead of at build. dataos-train reads the model's own
            # model_type and resolves the domain; this side only expands the declaration.
            families = tool.get("model_families") or []
            for family in families:
                env_backends[f"{b}@{family}"] = tool.get("path")
            if not families:
                env_backends[b] = tool.get("path")
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
            # Run first, environment second — the same precedence as `nodelist` below, and the
            # third submitter to learn it. It read env-only, and when `qos` moved to the run
            # config the field it read simply vanished: training jobs went out with no --qos at
            # all and Slurm put them on the account default, which is the GUARDED queue, while
            # their own prompt told them they were on an unguarded one. Nothing errored.
            "qos": resource.get("qos") or env.get("slurm", {}).get("qos"),
            # run pin wins; else the env's node pool (env.slurm.nodes)
            "nodelist": resource.get("node") or env.get("slurm", {}).get("nodes"),
            "gres": f"gpu:{gpus}",
            "cpus_per_task": cpus,
        },
    }
