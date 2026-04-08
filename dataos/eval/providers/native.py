from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.eval.providers.base import EvalRunContext
from dataos.eval.registry import get_eval_adapter, get_eval_task
from dataos.eval.types import EvalRunOutput


class NativeEvalProvider:
    name = "native"

    def run(self, context: EvalRunContext) -> EvalRunOutput:
        dataset_artifact = context.dataset_artifact
        if dataset_artifact is None:
            raise ValueError("eval_model requires at least one DatasetArtifact input")
        if context.config.task is None:
            raise ValueError("EvalConfig.task is required when provider='native'")
        if context.config.adapter is None:
            raise ValueError("EvalConfig.adapter is required when provider='native'")
        if context.config.request is None:
            raise ValueError("EvalConfig.request is required when provider='native'")

        task = get_eval_task(context.config.task.name)
        request_type = context.config.request.type
        if request_type not in task.supported_request_types:
            supported = ", ".join(sorted(task.supported_request_types))
            raise ValueError(
                f"Eval task '{task.name}' does not support request type '{request_type}'. Supported: {supported}"
            )

        dataset_obj = dataset_artifact.load()
        requested_split = context.config.data.split or task.default_split()
        eval_dataset, eval_split = _select_eval_split(dataset_obj, requested_split)
        if context.config.data.max_samples is not None:
            limit = min(context.config.data.max_samples, eval_dataset.num_rows)
            eval_dataset = eval_dataset.select(range(limit))

        requests = task.build_requests(eval_dataset, context.config)
        adapter = get_eval_adapter(context.config.adapter.name)
        responses = _run_requests(adapter, context, requests)
        bundle = task.score_responses(eval_dataset, requests, responses, context.config)
        return EvalRunOutput(
            provider=self.name,
            task_name=task.name,
            task_version=task.version,
            adapter_name=adapter.name,
            request_type=request_type,
            dataset_split=eval_split,
            summary=bundle,
            requests=requests,
            responses=responses,
            sample_results=bundle.samples,
        )


def _select_eval_split(
    dataset_obj: Dataset | DatasetDict,
    requested_split: str,
) -> tuple[Dataset, str]:
    if isinstance(dataset_obj, DatasetDict):
        if requested_split not in dataset_obj:
            available = ", ".join(dataset_obj.keys())
            raise ValueError(f"Requested eval split '{requested_split}' not found. Available: {available}")
        return dataset_obj[requested_split], requested_split
    if requested_split != "train":
        raise ValueError("DatasetArtifact is a single Dataset; set eval_config.data.split to 'train'")
    return dataset_obj, "train"


def _run_requests(adapter, context: EvalRunContext, requests):
    request_type = context.config.request.type
    if request_type == "generate_until":
        return adapter.generate_until(context.model_artifact, requests, context.config)
    raise ValueError(f"Unsupported eval request type: {request_type}")
