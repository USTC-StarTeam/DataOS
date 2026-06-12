from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any

from dataos._redact import redact
from dataos.eval.config import EvalConfig
from dataos.eval.types import LmEvalRunOutput
from dataos.eval.types import ModelLike as ModelArtifact


def run_lm_eval_native(model_artifact: ModelArtifact, config: EvalConfig) -> LmEvalRunOutput:
    """Run lm-eval by delegating model loading/inference to lm-eval itself."""

    _register_dataos_lm_eval_extensions()

    from lm_eval import simple_evaluate

    model_name = config.model or "hf"
    model_args = _resolve_model_args(model_artifact, config, model_name=model_name)
    model_for_eval, model_args_for_eval = _resolve_model_for_eval(
        model_name,
        model_args,
        batch_size=config.batch_size,
        max_batch_size=config.max_batch_size,
        device=config.device,
    )
    seeds = _resolve_seeds(config)
    task_manager = _build_task_manager(config)
    call_kwargs: dict[str, Any] = {
        "model": model_for_eval,
        "model_args": model_args_for_eval,
        "tasks": list(config.tasks),
        "num_fewshot": config.num_fewshot,
        "batch_size": config.batch_size,
        "max_batch_size": config.max_batch_size,
        "device": config.device,
        "use_cache": config.use_cache,
        "cache_requests": config.cache_requests,
        "rewrite_requests_cache": config.rewrite_requests_cache,
        "delete_requests_cache": config.delete_requests_cache,
        "limit": config.limit,
        "samples": config.samples,
        "bootstrap_iters": config.bootstrap_iters,
        "check_integrity": config.check_integrity,
        "write_out": config.write_out,
        "log_samples": config.log_samples,
        "system_instruction": config.system_instruction,
        "apply_chat_template": config.apply_chat_template,
        "fewshot_as_multiturn": config.fewshot_as_multiturn,
        "gen_kwargs": config.gen_kwargs,
        "task_manager": task_manager,
        "predict_only": config.predict_only,
        "random_seed": seeds["random_seed"],
        "numpy_random_seed": seeds["numpy_random_seed"],
        "torch_random_seed": seeds["torch_random_seed"],
        "fewshot_random_seed": seeds["fewshot_random_seed"],
        "confirm_run_unsafe_code": config.confirm_run_unsafe_code,
        "metadata": dict(config.metadata),
    }
    # HF evaluate's code_eval metric (humaneval/mbpp) gates on os.environ["HF_ALLOW_CODE_EVAL"]
    # SEPARATELY from lm-eval's confirm_run_unsafe_code — without it the code benches raise.
    if config.confirm_run_unsafe_code:
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    raw_dict, skipped = _evaluate_with_isolation(simple_evaluate, call_kwargs, list(config.tasks))
    return LmEvalRunOutput(
        tasks=list(config.tasks),
        results=dict(raw_dict.get("results", {})),
        samples=dict(raw_dict.get("samples", {})),
        raw=raw_dict,
        protocol={
            "mode": "lm_eval_native",
            "lm_eval_model": model_name,
            "model_args": model_args,
            "include_path": config.include_path,
            "metadata": dict(config.metadata),
            "seeds": seeds,
            "skipped_tasks": skipped,
        },
    )


def _as_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    return raw if isinstance(raw, dict) else dict(raw)


def _evaluate_with_isolation(
    simple_evaluate: Any, call_kwargs: dict[str, Any], tasks: list[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run all tasks in one call; if that fails and there are several, degrade to per-task.

    lm-eval resolves every task's dataset up front (``get_task_dict``), so one gated or broken
    bench (e.g. a gated GPQA) aborts the whole suite. Falling back to per-task keeps the
    measurable benches and records what was skipped instead of voiding everything.
    """
    try:
        return _as_dict(simple_evaluate(**_supported_kwargs(simple_evaluate, call_kwargs))), {}
    except Exception:  # noqa: BLE001 - fall back to per-task isolation below
        if len(tasks) <= 1:
            raise
    merged_results: dict[str, Any] = {}
    merged_samples: dict[str, Any] = {}
    merged_raw: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for task in tasks:
        per_kwargs = dict(call_kwargs)
        per_kwargs["tasks"] = [task]
        try:
            one = _as_dict(simple_evaluate(**_supported_kwargs(simple_evaluate, per_kwargs)))
        except Exception as task_exc:  # noqa: BLE001 - skip the broken bench, keep the rest
            skipped[task] = redact(str(task_exc))  # error text reaches disk via the manifest
            continue
        merged_results.update(one.get("results", {}))
        merged_samples.update(one.get("samples", {}))
        for key, value in one.items():
            merged_raw.setdefault(key, value)
    if not merged_results:
        raise RuntimeError(f"all eval tasks failed: {skipped}")
    merged_raw["results"] = merged_results
    merged_raw["samples"] = merged_samples
    merged_raw["skipped_tasks"] = skipped
    return merged_raw, skipped


def _resolve_model_for_eval(
    model_name: str,
    model_args: dict[str, Any] | str,
    *,
    batch_size: int | str | None,
    max_batch_size: int | None,
    device: str | None,
) -> tuple[Any, dict[str, Any] | str | None]:
    if not isinstance(model_args, dict) or "chat_template" not in model_args:
        return model_name, model_args

    template = _load_chat_template(model_args["chat_template"])
    init_args = dict(model_args)
    init_args.pop("chat_template", None)

    from lm_eval.api.registry import get_model

    lm = get_model(model_name).create_from_arg_obj(
        init_args,
        {
            "batch_size": batch_size,
            "max_batch_size": max_batch_size,
            "device": device,
        },
    )
    tokenizer = getattr(lm, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.chat_template = template
    if hasattr(lm, "hf_chat_template"):
        lm.hf_chat_template = template
    return lm, None


def _load_chat_template(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model_args.chat_template must be a non-empty string path or template body")
    candidate = Path(value).expanduser()
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return value


def _supported_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    filtered = {key: value for key, value in kwargs.items() if value is not None}
    parameters = inspect.signature(function).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return filtered
    return {key: value for key, value in filtered.items() if key in parameters}


def _build_task_manager(config: EvalConfig) -> Any | None:
    if config.include_path is None:
        return None
    from lm_eval.tasks import TaskManager

    kwargs: dict[str, Any] = {"include_path": config.include_path}
    if config.metadata:
        kwargs["metadata"] = dict(config.metadata)
    try:
        return TaskManager(**kwargs)
    except TypeError:
        kwargs.pop("metadata", None)
        return TaskManager(**kwargs)


def _resolve_model_args(
    model_artifact: ModelArtifact,
    config: EvalConfig,
    *,
    model_name: str | None = None,
) -> dict[str, Any] | str:
    if isinstance(config.model_args, str):
        return _resolve_model_args_string(model_artifact, config, model_name=model_name)

    model_args = dict(config.model_args or {})
    if "pretrained" not in model_args:
        model_args["pretrained"] = _default_pretrained_source(model_artifact)

    if _looks_like_lora_adapter(model_artifact):
        lora_key = _lora_model_arg_key(model_name)
        if lora_key not in model_args and "peft" not in model_args and "lora_local_path" not in model_args:
            model_args[lora_key] = str(model_artifact.path)

    if config.trust_remote_code is not None:
        model_args.setdefault("trust_remote_code", bool(config.trust_remote_code))
    return model_args


def _resolve_model_args_string(
    model_artifact: ModelArtifact,
    config: EvalConfig,
    *,
    model_name: str | None = None,
) -> str:
    model_args = config.model_args.strip() if isinstance(config.model_args, str) else ""
    if "pretrained=" not in model_args:
        model_args = _append_model_arg(
            model_args,
            "pretrained",
            _default_pretrained_source(model_artifact),
        )
    if _looks_like_lora_adapter(model_artifact):
        lora_key = _lora_model_arg_key(model_name)
        if "peft=" not in model_args and "lora_local_path=" not in model_args:
            model_args = _append_model_arg(model_args, lora_key, str(model_artifact.path))
    if config.trust_remote_code is not None and "trust_remote_code=" not in model_args:
        model_args = _append_model_arg(
            model_args,
            "trust_remote_code",
            str(bool(config.trust_remote_code)),
        )
    return model_args


def _append_model_arg(model_args: str, key: str, value: str) -> str:
    rendered = f"{key}={value}"
    if not model_args:
        return rendered
    return f"{model_args},{rendered}"


def _lora_model_arg_key(model_name: str | None) -> str:
    if (model_name or "").strip().lower() == "vllm":
        return "lora_local_path"
    return "peft"


def _default_pretrained_source(model_artifact: ModelArtifact) -> str:
    if _looks_like_lora_adapter(model_artifact):
        base_model = model_artifact.meta.get("base_model")
        if base_model:
            return str(base_model)
    return str(model_artifact.path)


def _looks_like_lora_adapter(model_artifact: ModelArtifact) -> bool:
    if str(model_artifact.meta.get("finetuning_type", "")).lower() == "lora":
        return True
    return (Path(model_artifact.path) / "adapter_config.json").exists()


def _register_dataos_lm_eval_extensions() -> None:
    """Register DataOS-owned lm-eval model backends before resolution."""
    import dataos.eval.models  # noqa: F401


def _resolve_seeds(config: EvalConfig) -> dict[str, int | None]:
    if config.seed is None:
        return {
            "random_seed": config.random_seed,
            "numpy_random_seed": config.numpy_random_seed,
            "torch_random_seed": config.torch_random_seed,
            "fewshot_random_seed": config.fewshot_random_seed,
        }
    if isinstance(config.seed, int):
        return {
            "random_seed": config.seed,
            "numpy_random_seed": config.seed,
            "torch_random_seed": config.seed,
            "fewshot_random_seed": config.seed,
        }
    values = list(config.seed)
    if len(values) != 4:
        raise ValueError("EvalConfig.seed must be an int or a 4-item seed list")
    return {
        "random_seed": values[0],
        "numpy_random_seed": values[1],
        "torch_random_seed": values[2],
        "fewshot_random_seed": values[3],
    }
