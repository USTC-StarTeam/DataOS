"""Machine-readable schemas for agent-side TrainSpec authoring.

``capabilities`` answers which backends/methods exist. This module answers what
shape a planner is allowed to emit for those methods, using the same Pydantic
models that validate runtime input.
"""

from __future__ import annotations

from typing import Any

from .registry import backend_names, get_backend
from .spec import (
    DPO,
    GRPO,
    KTO,
    ORPO,
    PPO,
    SFT,
    DatasetRef,
    EnvSpec,
    OnlineOpSpec,
    Reward,
    Rollout,
    TrainSpec,
    VerlPPO,
)
from .types import Capabilities, Handle, JobSpec, LaunchPlan, Resources, TrainResult

_METHOD_SCHEMAS = {
    "sft": SFT,
    "dpo": DPO,
    "ppo": PPO,
    "kto": KTO,
    "orpo": ORPO,
    "grpo": GRPO,
    "verl_ppo": VerlPPO,
}

_NAMED_SCHEMAS = {
    "trainspec": TrainSpec,
    "train_spec": TrainSpec,
    "dataset": DatasetRef,
    "env": EnvSpec,
    "online_op": OnlineOpSpec,
    "sft": SFT,
    "dpo": DPO,
    "ppo": PPO,
    "kto": KTO,
    "orpo": ORPO,
    "grpo": GRPO,
    "verl_ppo": VerlPPO,
    "rollout": Rollout,
    "reward": Reward,
    "resources": Resources,
    "launch": LaunchPlan,
    "job": JobSpec,
    "handle": Handle,
    "result": TrainResult,
    "capabilities": Capabilities,
}

_RUNTIME_OWNED_FIELDS = [
    "base_model",
    "run_dir",
    "dataset.path",
    "env.conda_sh",
    "env.backends",
    "env.deploy_root",
    "cluster",
]

_RUNTIME_OWNED_NATIVE_KEYS = [
    "model_name_or_path",
    "dataset",
    "dataset_dir",
    "output_dir",
    "overwrite_output_dir",
    "run_train.sh",
    "sbatch script",
]

_REWARD_PROTOCOL = (
    "verl methods: method.reward.path is runtime-owned — an agent-written .py file defining "
    "compute_score(data_source, solution_str, ground_truth, **kw) -> float. Derive it from the "
    "task's frozen format contract and self-check against known (solution, ground_truth) pairs; "
    "this library ships no task rewards."
)

_ENV_DOMAIN_RULE = (
    "env.backends is keyed by env domain: the backend name by default, or the "
    "method-scoped key from Capabilities.env_domains (e.g. 'llamafactory:ppo') for "
    "methods whose dependency domain differs. The runtime must provide the domain key "
    "that the chosen (backend, method) resolves to."
)


def method_names() -> list[str]:
    return sorted(_METHOD_SCHEMAS)


def schema_names() -> list[str]:
    return sorted(_NAMED_SCHEMAS)


def model_schema(name: str) -> dict[str, Any]:
    """Return the JSON schema for a named DataOS-train model or method."""
    key = name.lower().replace("-", "_")
    try:
        model = _NAMED_SCHEMAS[key]
    except KeyError:
        known = ", ".join(schema_names())
        raise KeyError(f"unknown schema {name!r}; known: {known}") from None
    return model.model_json_schema()


def method_schema(method: str) -> dict[str, Any]:
    """Return the JSON schema for a train method such as ``sft`` or ``grpo``."""
    key = method.lower().replace("-", "_")
    try:
        model = _METHOD_SCHEMAS[key]
    except KeyError:
        known = ", ".join(method_names())
        raise KeyError(f"unknown method {method!r}; known: {known}") from None
    return model.model_json_schema()


def backend_schema(backend: str) -> dict[str, Any]:
    """Return backend capabilities plus schemas for every method it supports."""
    be = get_backend(backend)
    caps = be.capabilities()
    return {
        "backend": caps.model_dump(),
        "methods": {
            method: method_schema(method) for method in caps.methods if method in _METHOD_SCHEMAS
        },
        "agent_contract": {
            "planner_may_set": [
                "name",
                "backend",
                "method",
                "resources",
                "online_ops",
                "overrides",
            ],
            "runtime_owned_fields": _RUNTIME_OWNED_FIELDS,
            "runtime_owned_native_keys": _RUNTIME_OWNED_NATIVE_KEYS,
            "env_domains": _ENV_DOMAIN_RULE,
            "reward_protocol": _REWARD_PROTOCOL,
        },
    }


def backend_method_schema(backend: str, method: str) -> dict[str, Any]:
    """Return one method schema scoped to a backend, rejecting unsupported pairs."""
    caps = get_backend(backend).capabilities()
    if method not in caps.methods:
        known = ", ".join(caps.methods)
        raise KeyError(f"backend {backend!r} does not support method {method!r}; known: {known}")
    return {
        "backend": caps.model_dump(),
        "method": method,
        "schema": method_schema(method),
    }


def all_schemas() -> dict[str, Any]:
    """Return the full schema bundle intended for upstream agent discovery."""
    return {
        "backends": {name: backend_schema(name) for name in backend_names()},
        "train_spec": TrainSpec.model_json_schema(),
        "supporting": {
            "dataset": DatasetRef.model_json_schema(),
            "env": EnvSpec.model_json_schema(),
            "online_op": OnlineOpSpec.model_json_schema(),
            "resources": Resources.model_json_schema(),
        },
        "agent_contract": {
            "planner_may_set": [
                "TrainSpec.name",
                "TrainSpec.backend",
                "TrainSpec.method",
                "TrainSpec.resources",
                "TrainSpec.online_ops",
                "TrainSpec.overrides",
            ],
            "runtime_owned_fields": _RUNTIME_OWNED_FIELDS,
            "runtime_owned_native_keys": _RUNTIME_OWNED_NATIVE_KEYS,
            "env_domains": _ENV_DOMAIN_RULE,
            "reward_protocol": _REWARD_PROTOCOL,
            "validation_gate": "Run `dataos-train validate <spec.json>` before build/submit.",
        },
    }
