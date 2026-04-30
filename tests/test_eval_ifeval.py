from __future__ import annotations

from datasets import Dataset, DatasetDict

from dataos.eval.tasks.ifeval import IFEvalTask
from dataos.eval.types import (
    EvalAdapterConfig,
    EvalConfig,
    EvalDataConfig,
    EvalRequestConfig,
    EvalTaskConfig,
    ModelResponse,
)


def _build_dataset() -> DatasetDict:
    rows = {
        "key": ["sample-1", "sample-2"],
        "prompt": [
            "Write a one-paragraph note ending with the phrase Have a great day.",
            "Reply in lowercase only and avoid the word secret.",
        ],
        "instruction_id_list": [
            ["startend:end_checker", "length_constraints:number_words"],
            ["change_case:english_lowercase", "keywords:forbidden_words"],
        ],
        "kwargs": [
            [
                {"end_phrase": "Have a great day."},
                {"relation": "at least", "num_words": 5},
            ],
            [
                {},
                {"forbidden_words": ["secret"]},
            ],
        ],
    }
    return DatasetDict({"test": Dataset.from_dict(rows)})


def _config() -> EvalConfig:
    return EvalConfig(
        provider="native",
        task=EvalTaskConfig(name="ifeval"),
        data=EvalDataConfig(split="test"),
        adapter=EvalAdapterConfig(name="transformers"),
        request=EvalRequestConfig(type="generate_until"),
    )


def test_ifeval_scores_prompt_and_instruction_accuracy():
    task = IFEvalTask()
    dataset = _build_dataset()["test"]
    config = _config()
    requests = task.build_requests(dataset, config)
    assert [r.sample_id for r in requests] == ["sample-1", "sample-2"]

    responses = [
        ModelResponse(
            sample_id="sample-1",
            index=0,
            request_type="generate_until",
            text="This note is short but warm. Have a great day.",
        ),
        ModelResponse(
            sample_id="sample-2",
            index=1,
            request_type="generate_until",
            text="this reply contains the word secret",
        ),
    ]

    bundle = task.score_responses(dataset, requests, responses, config)

    assert bundle.metric_name == "prompt_strict_accuracy"
    assert bundle.summary["num_examples"] == 2
    assert bundle.summary["instruction_total"] == 4
    assert bundle.summary["prompt_strict_accuracy"] == 0.5
    assert bundle.summary["instruction_strict_accuracy"] == 0.75

    sample_one = bundle.samples[0]
    assert sample_one.metrics["prompt_followed"] == 1.0
    assert all(v["followed"] for v in sample_one.meta["verdicts"])

    sample_two = bundle.samples[1]
    assert sample_two.metrics["prompt_followed"] == 0.0
    verdicts = {v["instruction_id"]: v for v in sample_two.meta["verdicts"]}
    assert verdicts["change_case:english_lowercase"]["followed"] is True
    assert verdicts["keywords:forbidden_words"]["followed"] is False


def test_ifeval_skips_unknown_instruction_ids():
    task = IFEvalTask()
    dataset = Dataset.from_dict(
        {
            "key": ["sample-x"],
            "prompt": ["Do something."],
            "instruction_id_list": [["this_does_not_exist:anywhere"]],
            "kwargs": [[{}]],
        }
    )
    config = _config()
    requests = task.build_requests(dataset, config)
    responses = [
        ModelResponse(
            sample_id="sample-x",
            index=0,
            request_type="generate_until",
            text="ok",
        )
    ]

    bundle = task.score_responses(dataset, requests, responses, config)
    assert bundle.summary["skipped_unregistered_ids"] == ["this_does_not_exist:anywhere"]
    assert bundle.summary["prompt_strict_accuracy"] == 0.0
