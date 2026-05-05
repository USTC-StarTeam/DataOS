"""Drive the lm-evaluation-harness bridge from the runtime stage.

``run_eval_stage`` is the single entry point. It builds an
:class:`EvalConfig` from ``task.eval_config``, picks the latest
``ModelArtifact`` from upstream artifacts, instantiates the requested
DataOS adapter, and calls :func:`run_lm_eval`. The :class:`LmEvalRunOutput`
is then packaged into one :class:`EvalArtifact` whose meta carries the
per-task results, the headline score, and a budget-trimmed copy of the
raw lm-eval payload.
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
    config = EvalConfig(**task.eval_config)
    model = _latest_model_artifact(artifacts)
    adapter = _make_adapter(config.adapter, config.adapter_config)
    output = _run_lm_eval(model, adapter, config)
    artifact = _eval_output_to_artifact(output, sandbox, model, config)
    sandbox.log("eval_model", _build_log_message(model, artifact, output, config))
    return [artifact]


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
    summary: dict[str, Any] = {
        "backend": "lm_eval",
        "tasks": list(config.tasks),
        "adapter": config.adapter,
        "limit": config.limit,
        "num_fewshot": config.num_fewshot,
        "batch_size": config.batch_size,
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
    keep = {"results", "configs", "versions", "n-shot", "task_hashes", "model_source"}
    trimmed = {key: value for key, value in raw.items() if key in keep}
    if len(trimmed) < min(len(raw), _RAW_TRIM_MAX_KEYS):
        for key in list(raw.keys())[: _RAW_TRIM_MAX_KEYS - len(trimmed)]:
            if key not in trimmed and key != "samples":
                trimmed[key] = raw[key]
    return trimmed


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
        f"tasks={','.join(config.tasks)}",
        f"adapter={config.adapter}",
        f"score={headline if headline is not None else 'n/a'}",
    ]
    if config.limit is not None:
        lines.append(f"limit={config.limit}")
    return "\n".join(lines) + "\n"
