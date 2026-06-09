"""Drive the lm-evaluation-harness bridge from the eval stage.

``run_eval_stage`` is the single entry point. It builds an
:class:`EvalConfig` from the open ``config`` dict and routes either to the
original DataOSLM adapter bridge or to lm-eval's native model backends.
The :class:`LmEvalRunOutput` is then packaged into one :class:`Eval`
whose meta carries the per-task results, the headline score, a protocol
snapshot, and a budget-trimmed copy of the raw lm-eval payload. Legacy
``provider="stub"`` configs still emit a local stub eval for smoke tests and
dry runs.

The model argument is any :class:`dataos.experiment.Model` (structurally a
``ModelLike``): the runner only reads ``.path`` / ``.meta`` / ``.artifact_id``
/ ``.producer``.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from typing import TYPE_CHECKING, Any

from dataos.eval.config import EvalConfig
from dataos._artifact import Eval, Model
from dataos._ids import _make_id

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataos.workspace import Workspace

_RAW_TRIM_MAX_KEYS = 50


def run_eval_stage(
    model: Model,
    config: dict[str, Any] | None,
    *,
    workspace: "Workspace",
) -> Eval:
    """Run an lm-eval task suite and emit a single :class:`Eval`."""
    eval_config = dict(config or {})
    if eval_config.get("provider") == "stub":
        evaluation = _stub_eval_artifact(workspace, model, eval_config)
        workspace.log(
            "eval_model",
            _build_stub_log_message(model, evaluation, eval_config),
        )
        return evaluation
    eval_config.pop("provider", None)
    eval_config.pop("provider_config", None)
    # eval config maps to EvalConfig's fields (a broad but fixed set). Tolerate unknown keys
    # instead of crashing with a raw TypeError — drop them with a warning.
    known = {f.name for f in dataclasses.fields(EvalConfig)}
    unknown = sorted(set(eval_config) - known)
    if unknown:
        warnings.warn(f"eval: ignoring config keys not in EvalConfig: {unknown}", stacklevel=2)
        eval_config = {k: v for k, v in eval_config.items() if k in known}
    config_obj = EvalConfig(**eval_config)
    mode = config_obj.normalized_mode()
    if mode == "lm_eval_native":
        output = _run_lm_eval_native(model, config_obj)
    else:
        adapter = _make_adapter(config_obj.adapter, config_obj.adapter_config)
        output = _run_lm_eval(model, adapter, config_obj)
    evaluation = _eval_output_to_artifact(output, workspace, model, config_obj)
    workspace.log("eval_model", _build_log_message(model, evaluation, output, config_obj))
    return evaluation


def _stub_eval_artifact(
    workspace: "Workspace",
    model: Model,
    eval_config: dict[str, Any],
) -> Eval:
    provider_config = dict(eval_config.get("provider_config", {}))
    score = float(provider_config.get("score", eval_config.get("score", 0.0)))
    eval_id = _make_id("eval")
    eval_dir = workspace.artifact_dir("eval", eval_id)
    summary: dict[str, Any] = {
        "backend": "stub",
        "provider": "stub",
        "score": score,
        "headline_score": score,
        "source_model_id": model.artifact_id,
        "base_model": model.meta.get("base_model"),
        "provider_config": provider_config,
    }
    (eval_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return Eval(
        eval_id=eval_id,
        path=eval_dir,
        metrics={},
        score=score,
        meta=summary,
        producer="eval_model",
    )


def _make_adapter(name: str, adapter_config: dict[str, Any]) -> Any:
    del adapter_config  # adapters read their knobs off AdapterCallContext at call time
    if name == "transformers":
        from dataos.eval.adapters.transformers import TransformersAdapter

        return TransformersAdapter()
    if name == "openai_compatible":
        from dataos.eval.adapters.openai_compatible import OpenAICompatibleAdapter

        return OpenAICompatibleAdapter()
    raise ValueError(f"Unknown eval adapter {name!r}; expected 'transformers' or 'openai_compatible'")


def _run_lm_eval(model: Model, adapter: Any, config: EvalConfig):
    from dataos.eval.backend import run_lm_eval

    # Forward the seed hierarchy to lm-eval so the legacy-adapter path is reproducible too
    # (native mode already does this; this closes the gap).
    seeds = {
        "random_seed": config.random_seed,
        "numpy_random_seed": config.numpy_random_seed,
        "torch_random_seed": config.torch_random_seed,
        "fewshot_random_seed": config.fewshot_random_seed,
    }
    return run_lm_eval(
        model,
        adapter,
        tasks=list(config.tasks),
        limit=config.limit,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        max_new_tokens=config.max_new_tokens,
        apply_chat_template=config.apply_chat_template,
        confirm_run_unsafe_code=config.confirm_run_unsafe_code,
        adapter_config=dict(config.adapter_config),
        extra_simple_evaluate_kwargs=seeds,
    )


def _run_lm_eval_native(model: Model, config: EvalConfig):
    from dataos.eval.backend import run_lm_eval_native

    return run_lm_eval_native(model, config)


def _eval_seeds(config: EvalConfig) -> dict[str, Any]:
    """The effective seed hierarchy lm-eval ran with (EvalConfig defaults if user didn't set)."""
    return {
        "seed": config.seed,
        "random_seed": config.random_seed,
        "numpy_random_seed": config.numpy_random_seed,
        "torch_random_seed": config.torch_random_seed,
        "fewshot_random_seed": config.fewshot_random_seed,
    }


def _eval_output_to_artifact(
    output: Any,
    workspace: "Workspace",
    model: Model,
    config: EvalConfig,
) -> Eval:
    eval_id = _make_id("eval")
    eval_dir = workspace.artifact_dir("eval", eval_id)
    metrics = _flatten_metrics(output.results)
    headline = _pick_headline(metrics, config.tasks, config.headline_metric)
    protocol = _protocol_snapshot(output, model, config)
    summary: dict[str, Any] = {
        "backend": "lm_eval",
        "mode": config.normalized_mode(),
        "tasks": list(config.tasks),
        "limit": config.limit,
        "num_fewshot": config.num_fewshot,
        "batch_size": config.batch_size,
        "model": protocol["resolved"].get("lm_eval_model"),
        "adapter": config.adapter if config.normalized_mode() == "legacy_adapter" else None,
        "results": dict(output.results),
        "score": headline,
        "headline_score": headline,
        "source_model_id": model.artifact_id,
        "base_model": model.meta.get("base_model"),
        "seeds": _eval_seeds(config),  # effective seeds (incl. EvalConfig defaults) for repro
    }
    (eval_dir / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (eval_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    raw = _trim_raw(output.raw)
    (eval_dir / "lm_eval_results.json").write_text(
        json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if output.samples:
        for task_name, samples in output.samples.items():
            safe = task_name.replace("/", "_")
            with (eval_dir / f"samples-{safe}.jsonl").open("w", encoding="utf-8") as handle:
                for record in samples:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return Eval(
        eval_id=eval_id,
        path=eval_dir,
        metrics=metrics,
        score=headline,
        meta=summary,
        producer="eval_model",
    )


def _flatten_metrics(results: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for task_name, metrics in (results or {}).items():
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and "stderr" not in key and key != "alias":
                flat[f"{task_name}::{key}"] = float(value)
    return flat


def _pick_headline(
    metrics: dict[str, float], tasks: list[str], headline_metric: str | None = None
) -> float | None:
    # A task can emit several metrics (e.g. gsm8k -> strict-match AND flexible-extract). If the
    # caller pinned one via config["headline_metric"], honor it; otherwise fall back to the first.
    if headline_metric:
        for task in tasks:
            key = f"{task}::{headline_metric}"
            if key in metrics:
                return metrics[key]
    for task in tasks:
        for key, value in metrics.items():
            if key.startswith(f"{task}::"):
                return value
    return None


def _trim_raw(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"_note": "raw payload was not a dict", "_type": type(raw).__name__}
    keep = {
        "results",
        "configs",
        "config",
        "versions",
        "n-shot",
        "task_hashes",
        "model_source",
        "model_configs",
        "higher_is_better",
        "git_hash",
        "date",
    }
    trimmed = {key: value for key, value in raw.items() if key in keep}
    if len(trimmed) < min(len(raw), _RAW_TRIM_MAX_KEYS):
        for key in list(raw.keys())[: _RAW_TRIM_MAX_KEYS - len(trimmed)]:
            if key not in trimmed and key != "samples":
                trimmed[key] = raw[key]
    return trimmed


def _protocol_snapshot(output: Any, model: Model, config: EvalConfig) -> dict[str, Any]:
    return {
        "eval_config": config.to_record(),
        "source_model": {
            "artifact_id": model.artifact_id,
            "path": str(model.path),
            "base_model": model.meta.get("base_model"),
            "finetuning_type": model.meta.get("finetuning_type"),
            "producer": model.producer,
        },
        "resolved": dict(getattr(output, "protocol", {}) or {}),
    }


def _build_log_message(
    model: Model,
    evaluation: Eval,
    output: Any,
    config: EvalConfig,
) -> str:
    headline = evaluation.meta.get("headline_score")
    lines = [
        f"model={model.artifact_id}",
        f"eval={evaluation.artifact_id}",
        "backend=lm_eval",
        f"mode={config.normalized_mode()}",
        f"tasks={','.join(config.tasks)}",
        f"score={headline if headline is not None else 'n/a'}",
    ]
    if config.normalized_mode() == "lm_eval_native":
        lines.append(f"lm_eval_model={evaluation.meta.get('model') or config.model or 'hf'}")
    else:
        lines.append(f"adapter={config.adapter}")
    if config.limit is not None:
        lines.append(f"limit={config.limit}")
    return "\n".join(lines) + "\n"


def _build_stub_log_message(
    model: Model,
    evaluation: Eval,
    eval_config: dict[str, Any],
) -> str:
    score = evaluation.meta.get("score")
    lines = [
        f"model={model.artifact_id}",
        f"eval={evaluation.artifact_id}",
        "backend=stub",
        f"score={score if score is not None else 'n/a'}",
    ]
    if eval_config.get("provider_config"):
        lines.append(f"provider_config={json.dumps(eval_config['provider_config'], sort_keys=True)}")
    return "\n".join(lines) + "\n"
