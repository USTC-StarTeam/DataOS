"""Drive the lm-evaluation-harness bridge from the runtime stage.

``run_eval_stage`` is the single entry point. It builds an
:class:`EvalConfig` from ``task.eval_config`` and routes either to the
original DataOSLM adapter bridge or to lm-eval's native model backends.
The :class:`LmEvalRunOutput` is then packaged into one :class:`EvalArtifact`
whose meta carries the per-task results, the headline score, a protocol
snapshot, and a budget-trimmed copy of the raw lm-eval payload. Legacy
``provider="stub"`` configs still emit a local stub eval artifact for smoke
tests and dry runs.
"""

from __future__ import annotations

import json
from typing import Any

from dataos.core.artifact import Artifact, EvalArtifact, ModelArtifact, make_artifact_id
from dataos.core.sandbox import Sandbox
from dataos.eval.config import EvalConfig

_RAW_TRIM_MAX_KEYS = 50


def run_eval_stage(
    artifacts: list[Artifact],
    sandbox: Sandbox,
    *,
    task: Any,
    experiment: Any | None = None,
    **_unused: Any,
) -> list[EvalArtifact]:
    """Run an lm-eval task suite and emit a single :class:`EvalArtifact`."""
    del experiment, _unused
    eval_config = dict(task.eval_config)
    model = _latest_model_artifact(artifacts)
    if eval_config.get("provider") == "stub":
        artifact = _stub_eval_artifact(sandbox, model, eval_config)
        sandbox.log(
            "eval_model",
            _build_stub_log_message(model, artifact, eval_config),
        )
        return [artifact]
    config = EvalConfig(**eval_config)
    mode = config.normalized_mode()
    if mode == "lm_eval_native":
        output = _run_lm_eval_native(model, config)
    else:
        adapter = _make_adapter(config.adapter, config.adapter_config)
        output = _run_lm_eval(model, adapter, config)
    artifact = _eval_output_to_artifact(output, sandbox, model, config)
    sandbox.log("eval_model", _build_log_message(model, artifact, output, config))
    return [artifact]


def _stub_eval_artifact(
    sandbox: Sandbox,
    model: ModelArtifact,
    eval_config: dict[str, Any],
) -> EvalArtifact:
    provider_config = dict(eval_config.get("provider_config", {}))
    score = float(provider_config.get("score", eval_config.get("score", 0.0)))
    artifact_id = make_artifact_id("eval")
    eval_dir = sandbox.artifact_dir("eval", artifact_id)
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
    return EvalArtifact(
        artifact_id=artifact_id,
        name=f"eval-{model.artifact_id}",
        path=eval_dir,
        parent_ids=[model.artifact_id],
        producer="eval_model",
        meta=summary,
    )


def _latest_model_artifact(artifacts: list[Artifact]) -> ModelArtifact:
    for artifact in reversed(artifacts):
        if isinstance(artifact, ModelArtifact):
            return artifact
    raise ValueError("run_eval_stage requires at least one ModelArtifact input")


def _make_adapter(name: str, adapter_config: dict[str, Any]) -> Any:
    del adapter_config  # adapters read their knobs off AdapterCallContext at call time
    if name == "transformers":
        from dataos.eval.adapters.transformers import TransformersAdapter

        return TransformersAdapter()
    if name == "openai_compatible":
        from dataos.eval.adapters.openai_compatible import OpenAICompatibleAdapter

        return OpenAICompatibleAdapter()
    raise ValueError(f"Unknown eval adapter {name!r}; expected 'transformers' or 'openai_compatible'")


def _run_lm_eval(model: ModelArtifact, adapter: Any, config: EvalConfig):
    from dataos.eval.backend import run_lm_eval

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
    )


def _run_lm_eval_native(model: ModelArtifact, config: EvalConfig):
    from dataos.eval.backend import run_lm_eval_native

    return run_lm_eval_native(model, config)


def _eval_output_to_artifact(
    output: Any,
    sandbox: Sandbox,
    model: ModelArtifact,
    config: EvalConfig,
) -> EvalArtifact:
    artifact_id = make_artifact_id("eval")
    eval_dir = sandbox.artifact_dir("eval", artifact_id)
    metrics = _flatten_metrics(output.results)
    headline = _pick_headline(metrics, config.tasks)
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
    return EvalArtifact(
        artifact_id=artifact_id,
        name=f"eval-{model.artifact_id}",
        path=eval_dir,
        parent_ids=[model.artifact_id],
        producer="eval_model",
        meta=summary,
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


def _pick_headline(metrics: dict[str, float], tasks: list[str]) -> float | None:
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


def _protocol_snapshot(output: Any, model: ModelArtifact, config: EvalConfig) -> dict[str, Any]:
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
    model: ModelArtifact,
    artifact: EvalArtifact,
    output: Any,
    config: EvalConfig,
) -> str:
    headline = artifact.meta.get("headline_score")
    lines = [
        f"model={model.artifact_id}",
        f"eval={artifact.artifact_id}",
        "backend=lm_eval",
        f"mode={config.normalized_mode()}",
        f"tasks={','.join(config.tasks)}",
        f"score={headline if headline is not None else 'n/a'}",
    ]
    if config.normalized_mode() == "lm_eval_native":
        lines.append(f"lm_eval_model={artifact.meta.get('model') or config.model or 'hf'}")
    else:
        lines.append(f"adapter={config.adapter}")
    if config.limit is not None:
        lines.append(f"limit={config.limit}")
    return "\n".join(lines) + "\n"


def _build_stub_log_message(
    model: ModelArtifact,
    artifact: EvalArtifact,
    eval_config: dict[str, Any],
) -> str:
    score = artifact.meta.get("score")
    lines = [
        f"model={model.artifact_id}",
        f"eval={artifact.artifact_id}",
        "backend=stub",
        f"score={score if score is not None else 'n/a'}",
    ]
    if eval_config.get("provider_config"):
        lines.append(f"provider_config={json.dumps(eval_config['provider_config'], sort_keys=True)}")
    return "\n".join(lines) + "\n"
