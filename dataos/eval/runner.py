from __future__ import annotations

from typing import Any

from dataos.core.artifact import Artifact, DatasetArtifact, ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.eval.config import resolve_eval_config
from dataos.eval.providers.base import EvalRunContext
from dataos.eval.registry import get_eval_provider
from dataos.eval.store import EvalArtifactStore
from dataos.eval.types import EvalRunOutput


def run_eval_stage(inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> list[Artifact]:
    task = kwargs.get("task")
    model_artifact = _find_latest_model(inputs)
    dataset_artifact = _find_latest_dataset(inputs)
    config = resolve_eval_config(getattr(task, "eval_config", {}), kwargs)
    provider = get_eval_provider(config.provider)
    output = provider.run(
        EvalRunContext(
            dataset_artifact=dataset_artifact,
            model_artifact=model_artifact,
            sandbox=sandbox,
            task_template=task,
            config=config,
        )
    )

    store = EvalArtifactStore(sandbox)
    artifacts: list[Artifact] = []
    trace_artifact_ids = {
        "requests": None,
        "responses": None,
        "samples": None,
    }
    request_artifact = None
    response_artifact = None
    sample_artifact = None

    base_trace_meta = _build_trace_meta(
        output=output,
        task=task,
        model_artifact=model_artifact,
        dataset_artifact=dataset_artifact,
    )
    request_records = [_request_record(request, output) for request in output.requests]
    response_records = [_response_record(response) for response in output.responses]
    sample_records = [_sample_record(sample) for sample in output.sample_results]

    if request_records and config.artifacts.persist_requests:
        request_parent_ids = [dataset_artifact.artifact_id] if dataset_artifact is not None else []
        request_artifact = store.save_requests(request_records, request_parent_ids, base_trace_meta)
        artifacts.append(request_artifact)
        trace_artifact_ids["requests"] = request_artifact.artifact_id
    if response_records and config.artifacts.persist_responses:
        response_parent_ids = _compact_parent_ids(
            [
                dataset_artifact.artifact_id if dataset_artifact is not None else None,
                model_artifact.artifact_id,
                request_artifact.artifact_id if request_artifact is not None else None,
            ]
        )
        response_artifact = store.save_responses(response_records, response_parent_ids, base_trace_meta)
        artifacts.append(response_artifact)
        trace_artifact_ids["responses"] = response_artifact.artifact_id
    if sample_records and config.artifacts.persist_samples:
        sample_parent_ids = _compact_parent_ids(
            [
                dataset_artifact.artifact_id if dataset_artifact is not None else None,
                model_artifact.artifact_id,
                response_artifact.artifact_id if response_artifact is not None else None,
            ]
        )
        sample_artifact = store.save_samples(sample_records, sample_parent_ids, base_trace_meta)
        artifacts.append(sample_artifact)
        trace_artifact_ids["samples"] = sample_artifact.artifact_id

    summary = _build_eval_summary(
        output=output,
        trace_artifact_ids=trace_artifact_ids,
        task=task,
        model_artifact=model_artifact,
        dataset_artifact=dataset_artifact,
    )
    eval_parent_ids = _compact_parent_ids(
        [
            dataset_artifact.artifact_id if dataset_artifact is not None else None,
            model_artifact.artifact_id,
            sample_artifact.artifact_id if sample_artifact is not None else None,
        ]
    )
    eval_artifact = store.save_eval_artifact(
        model_artifact=model_artifact,
        parent_ids=eval_parent_ids,
        summary=summary,
        sample_records=sample_records,
    )
    artifacts.append(eval_artifact)

    sandbox.log("eval_model", _build_log_message(model_artifact, dataset_artifact, eval_artifact, output))
    return artifacts


def _find_latest_model(inputs: list[Artifact]) -> ModelArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, ModelArtifact):
            return artifact
    raise ValueError("eval_model requires at least one ModelArtifact input")


def _find_latest_dataset(inputs: list[Artifact]) -> DatasetArtifact | None:
    for artifact in reversed(inputs):
        if isinstance(artifact, DatasetArtifact):
            return artifact
    return None


def _build_trace_meta(
    *,
    output: EvalRunOutput,
    task: Any,
    model_artifact: ModelArtifact,
    dataset_artifact: DatasetArtifact | None,
) -> dict[str, Any]:
    return {
        "provider": output.provider,
        "task": output.task_name,
        "task_version": output.task_version,
        "adapter": output.adapter_name,
        "request_type": output.request_type,
        "dataset_split": output.dataset_split,
        "source_dataset_id": dataset_artifact.artifact_id if dataset_artifact is not None else None,
        "source_model_id": model_artifact.artifact_id,
        "task_id": getattr(task, "task_id", None),
    }


def _build_eval_summary(
    *,
    output: EvalRunOutput,
    trace_artifact_ids: dict[str, str | None],
    task: Any,
    model_artifact: ModelArtifact,
    dataset_artifact: DatasetArtifact | None,
) -> dict[str, Any]:
    bundle = output.summary
    summary: dict[str, Any] = {
        "provider": output.provider,
        "task": output.task_name,
        "task_version": output.task_version,
        "adapter": output.adapter_name,
        "request_type": output.request_type,
        "metric": bundle.metric_name if bundle is not None else None,
        "score": float(bundle.score) if bundle is not None else 0.0,
        "dataset_split": output.dataset_split,
        "source_dataset_id": dataset_artifact.artifact_id if dataset_artifact is not None else None,
        "source_model_id": model_artifact.artifact_id,
        "task_id": getattr(task, "task_id", None),
        "base_model": model_artifact.meta.get("base_model"),
        "trace_artifact_ids": trace_artifact_ids,
    }
    if bundle is not None:
        summary.update(bundle.summary)
    return summary


def _request_record(request, output: EvalRunOutput) -> dict[str, Any]:
    record = request.to_record()
    record.update(
        {
            "task": output.task_name,
            "task_version": output.task_version,
        }
    )
    return record


def _response_record(response) -> dict[str, Any]:
    return response.to_record()


def _sample_record(sample) -> dict[str, Any]:
    record = sample.to_record()
    is_correct = None
    if "exact_match" in sample.metrics:
        is_correct = bool(sample.metrics["exact_match"])
    elif "correct" in sample.metrics:
        is_correct = bool(sample.metrics["correct"])
    record.update(
        {
            "predicted_answer": sample.normalized_prediction,
            "reference_answer": sample.normalized_reference,
            "is_correct": is_correct,
        }
    )
    return record


def _build_log_message(
    model_artifact: ModelArtifact,
    dataset_artifact: DatasetArtifact | None,
    eval_artifact: Artifact,
    output: EvalRunOutput,
) -> str:
    lines = [
        f"model={model_artifact.artifact_id}",
        f"eval={eval_artifact.artifact_id}",
        f"provider={output.provider}",
        f"score={float(output.summary.score if output.summary is not None else 0.0)}",
    ]
    if dataset_artifact is not None:
        lines.append(f"dataset={dataset_artifact.artifact_id}")
    if output.task_name:
        lines.append(f"task={output.task_name}")
    if output.dataset_split:
        lines.append(f"split={output.dataset_split}")
    return "\n".join(lines) + "\n"


def _compact_parent_ids(parent_ids: list[str | None]) -> list[str]:
    return [parent_id for parent_id in parent_ids if parent_id is not None]
