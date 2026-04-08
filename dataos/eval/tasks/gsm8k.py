from __future__ import annotations

from typing import Any

from datasets import Dataset

from dataos.eval.metrics.exact_match import aggregate_exact_match, exact_match
from dataos.eval.postprocessors.extract_final_answer import extract_final_answer
from dataos.eval.types import EvalConfig, EvalRequest, EvalResultBundle, EvalSampleResult, ModelResponse


class GSM8KTask:
    name = "gsm8k"
    version = "v1"
    supported_request_types = {"generate_until"}

    def default_split(self) -> str:
        return "test"

    def build_requests(self, dataset: Dataset, config: EvalConfig) -> list[EvalRequest]:
        del config
        requests: list[EvalRequest] = []
        for index, row in enumerate(dataset):
            instruction = row.get("instruction")
            if not isinstance(instruction, str) or not instruction:
                raise ValueError(f"Row {index} requires a non-empty 'instruction' field for eval")
            prompt = instruction
            row_input = row.get("input")
            if isinstance(row_input, str) and row_input.strip():
                prompt = f"{instruction}\n\n{row_input}"
            requests.append(
                EvalRequest(
                    sample_id=str(index),
                    index=index,
                    request_type="generate_until",
                    prompt=prompt,
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
        del config
        rows = list(dataset)
        if len(rows) != len(requests):
            raise ValueError("Request count does not match evaluation dataset size")
        if len(rows) != len(responses):
            raise ValueError("Response count does not match evaluation dataset size")

        samples: list[EvalSampleResult] = []
        sample_scores: list[float] = []
        for index, (row, request, response) in enumerate(zip(rows, requests, responses)):
            reference = _extract_reference_answer(row, index)
            normalized_prediction = extract_final_answer(response.text or "")
            normalized_reference = extract_final_answer(reference)
            score = exact_match(normalized_prediction, normalized_reference)
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
                        "instruction": row.get("instruction"),
                        "is_correct": bool(score),
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
            },
            samples=samples,
        )


def _extract_reference_answer(row: dict[str, Any], index: int) -> str:
    for field_name in ("final_answer", "answer", "response"):
        value = row.get(field_name)
        if isinstance(value, str) and value:
            return value
    raise ValueError(
        f"Row {index} requires one of 'final_answer', 'answer', or 'response' for gsm8k eval"
    )
