from __future__ import annotations

from typing import Any

from datasets import Dataset

from dataos.eval.metrics.exact_match import aggregate_exact_match
from dataos.eval.postprocessors.normalize_latex_answer import (
    canonical_latex_prediction_answer,
    canonical_latex_reference_answer,
    normalize_latex_prediction_answer_sequence,
    normalize_latex_reference_answer_sequence,
)
from dataos.eval.types import EvalConfig, EvalRequest, EvalResultBundle, EvalSampleResult, ModelResponse

DEFAULT_PROMPT_TEMPLATE = """{question}

Solve the problem. End your response with a separate line exactly in this format:
#### <answer>

Use the same mathematical form requested by the problem."""


class PhysicsTask:
    name = "physics"
    version = "v1"
    supported_request_types = {"generate_until"}

    def default_split(self) -> str:
        return "test"

    def build_requests(self, dataset: Dataset, config: EvalConfig) -> list[EvalRequest]:
        task_config = _task_config(config)
        question_field = str(task_config.get("question_field", "question"))
        prompt_template = str(task_config.get("prompt_template", DEFAULT_PROMPT_TEMPLATE))
        requests: list[EvalRequest] = []
        for index, row in enumerate(dataset):
            question = _extract_question(row, index=index, preferred_field=question_field)
            _extract_reference_answers(row, index=index, task_config=task_config)
            prompt_context = dict(row)
            prompt_context["question"] = question
            prompt = prompt_template.format(**prompt_context)
            requests.append(
                EvalRequest(
                    sample_id=str(index),
                    index=index,
                    request_type="generate_until",
                    prompt=prompt,
                    meta={
                        "question_field": question_field,
                    },
                )
            )
        return requests

    def score_responses(
        self,
        dataset: Dataset,
        requests: list[EvalRequest],
        responses: list[ModelResponse],
        config: EvalConfig,
    ) -> EvalResultBundle:
        rows = list(dataset)
        if len(rows) != len(requests):
            raise ValueError("Request count does not match evaluation dataset size")
        if len(rows) != len(responses):
            raise ValueError("Response count does not match evaluation dataset size")

        task_config = _task_config(config)
        samples: list[EvalSampleResult] = []
        sample_scores: list[float] = []
        invalid_predictions = 0
        for index, (row, request, response) in enumerate(zip(rows, requests, responses)):
            reference = _extract_reference_answers(row, index=index, task_config=task_config)
            normalized_reference = canonical_latex_reference_answer(reference)
            if normalized_reference is None:
                raise ValueError(f"Row {index} requires a normalizable answer for physics eval")
            normalized_prediction = canonical_latex_prediction_answer(response.text or "")
            invalid_predictions += int(normalized_prediction is None)
            score = float(
                normalized_prediction is not None and normalized_prediction == normalized_reference
            )
            sample_scores.append(score)
            samples.append(
                EvalSampleResult(
                    sample_id=request.sample_id,
                    index=index,
                    prompt=request.prompt,
                    prediction=response.text,
                    reference=reference,
                    normalized_prediction=normalized_prediction,
                    normalized_reference=normalized_reference,
                    metrics={"exact_match": score},
                    meta={
                        "is_correct": bool(score),
                        "is_invalid_prediction": normalized_prediction is None,
                        "normalized_prediction_sequence": normalize_latex_prediction_answer_sequence(
                            response.text or ""
                        ),
                        "normalized_reference_sequence": normalize_latex_reference_answer_sequence(
                            reference
                        ),
                    },
                )
            )

        total = len(samples)
        correct = int(sum(sample_scores))
        return EvalResultBundle(
            metric_name="exact_match",
            score=aggregate_exact_match(sample_scores),
            summary={
                "correct": correct,
                "num_examples": total,
                "invalid_predictions": invalid_predictions,
                "invalid_prediction_rate": invalid_predictions / total if total else 0.0,
                "matching_policy": "normalized_string_exact_match",
            },
            samples=samples,
        )


def _task_config(config: EvalConfig) -> dict[str, Any]:
    if config.task is None:
        return {}
    return dict(config.task.config)


def _extract_question(row: dict[str, Any], *, index: int, preferred_field: str) -> str:
    for field_name in (preferred_field, "problem", "instruction", "prompt", "input"):
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        f"Row {index} requires a non-empty question/problem/instruction field for physics eval"
    )


def _extract_reference_answers(row: dict[str, Any], *, index: int, task_config: dict[str, Any]) -> Any:
    answer_field = str(task_config.get("answer_field", "answer"))
    for field_name in (answer_field, "final_answer", "response"):
        value = row.get(field_name)
        if _has_value(value):
            return value
    raise ValueError(
        f"Row {index} requires one of '{answer_field}', 'final_answer', or 'response' "
        "for physics eval"
    )

def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True
