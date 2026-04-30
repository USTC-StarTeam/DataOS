from __future__ import annotations

from typing import Any

from dataos.eval.types import (
    EvalAdapterConfig,
    EvalArtifactPolicy,
    EvalConfig,
    EvalDataConfig,
    EvalRequestConfig,
    EvalTaskConfig,
)

ALLOWED_TOP_LEVEL_KEYS = {
    "adapter",
    "artifacts",
    "data",
    "provider",
    "provider_config",
    "request",
    "task",
}


def resolve_eval_config(task_eval_config: dict[str, Any], kwargs: dict[str, Any]) -> EvalConfig:
    if task_eval_config in ({}, None):
        task_eval_config = {}
    if not isinstance(task_eval_config, dict):
        raise ValueError("task.eval_config must be a dict")
    step_params = {key: value for key, value in kwargs.items() if key not in {"task", "experiment"}}
    return _resolve_eval_config(task_eval_config, step_params)


def _resolve_eval_config(task_eval_config: dict[str, Any], step_params: dict[str, Any]) -> EvalConfig:
    unknown_keys = set(task_eval_config) - ALLOWED_TOP_LEVEL_KEYS
    if unknown_keys:
        keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unsupported task.eval_config keys: {keys}")

    unknown_step_keys = set(step_params) - ALLOWED_TOP_LEVEL_KEYS
    if unknown_step_keys:
        keys = ", ".join(sorted(unknown_step_keys))
        raise ValueError(f"Unsupported eval_model step params: {keys}")

    merged = {
        key: _merge_config_value(task_eval_config.get(key), step_params.get(key))
        for key in ALLOWED_TOP_LEVEL_KEYS
    }
    provider = merged.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ValueError("eval_config.provider is required")

    provider_config = _ensure_dict(merged.get("provider_config", {}), "eval_config.provider_config")
    task = _resolve_task_config(merged.get("task"))
    data = _resolve_data_config(merged.get("data"))
    adapter = _resolve_adapter_config(merged.get("adapter"))
    request = _resolve_request_config(merged.get("request"))
    artifacts = _resolve_artifact_policy(merged.get("artifacts"))

    if provider == "native":
        if task is None:
            raise ValueError("eval_config.task is required when provider='native'")
        if adapter is None:
            raise ValueError("eval_config.adapter is required when provider='native'")
        if request is None:
            raise ValueError("eval_config.request is required when provider='native'")
    return EvalConfig(
        provider=provider,
        provider_config=provider_config,
        task=task,
        data=data,
        adapter=adapter,
        request=request,
        artifacts=artifacts,
    )


def _merge_config_value(base: Any, override: Any) -> Any:
    if override is None:
        return base
    if base is None:
        return override
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _merge_config_value(merged.get(key), value)
        return merged
    return override


def _ensure_dict(value: Any, owner: str) -> dict[str, Any]:
    if value in ({}, None):
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{owner} must be a dict")
    return dict(value)


def _resolve_task_config(value: Any) -> EvalTaskConfig | None:
    if value in ({}, None):
        return None
    payload = _ensure_dict(value, "eval_config.task")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("eval_config.task.name is required")
    config = _ensure_dict(payload.get("config", {}), "eval_config.task.config")
    return EvalTaskConfig(name=name, config=config)


def _resolve_data_config(value: Any) -> EvalDataConfig:
    payload = _ensure_dict(value, "eval_config.data")
    split = payload.get("split")
    if split is not None:
        split = str(split)
    max_samples = payload.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("eval_config.data.max_samples must be > 0 when provided")
    path = payload.get("path")
    if path is not None:
        path = str(path)
    return EvalDataConfig(split=split, max_samples=max_samples, path=path)


def _resolve_adapter_config(value: Any) -> EvalAdapterConfig | None:
    if value in ({}, None):
        return None
    payload = _ensure_dict(value, "eval_config.adapter")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("eval_config.adapter.name is required")
    config = _ensure_dict(payload.get("config", {}), "eval_config.adapter.config")
    return EvalAdapterConfig(name=name, config=config)


def _resolve_request_config(value: Any) -> EvalRequestConfig | None:
    if value in ({}, None):
        return None
    payload = _ensure_dict(value, "eval_config.request")
    request_type = payload.get("type")
    if not isinstance(request_type, str) or not request_type:
        raise ValueError("eval_config.request.type is required")
    batch_size = int(payload.get("batch_size", 8))
    max_new_tokens = int(payload.get("max_new_tokens", 256))
    do_sample = bool(payload.get("do_sample", False))
    if batch_size <= 0:
        raise ValueError("eval_config.request.batch_size must be > 0")
    if max_new_tokens <= 0:
        raise ValueError("eval_config.request.max_new_tokens must be > 0")
    return EvalRequestConfig(
        type=request_type,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )


def _resolve_artifact_policy(value: Any) -> EvalArtifactPolicy:
    payload = _ensure_dict(value, "eval_config.artifacts")
    return EvalArtifactPolicy(
        persist_requests=bool(payload.get("persist_requests", False)),
        persist_responses=bool(payload.get("persist_responses", False)),
        persist_samples=bool(payload.get("persist_samples", True)),
    )
