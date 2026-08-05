from __future__ import annotations

from dataos_operator.operators.evaluator.llm_task_relevance_evaluator import LLMTaskRelevanceEvaluator


def test_llm_task_relevance_evaluator_writes_normalized_score(monkeypatch) -> None:
    def fake_call_llm_json(**_: object) -> tuple[dict[str, object], str]:
        return (
            {
                "score": 5,
                "label": "relevant",
                "reason": "Matches the target task.",
            },
            '{"score": 5, "label": "relevant", "reason": "Matches the target task."}',
        )

    monkeypatch.setattr(
        "dataos_operator.operators.evaluator.llm_task_relevance_evaluator.call_llm_json",
        fake_call_llm_json,
    )

    result = LLMTaskRelevanceEvaluator(
        model="fake-model",
        task_desc="Math reasoning supervised fine-tuning data.",
    ).run(
        {
            "query": "What is 2 + 2?",
            "answer": "4",
        }
    )

    assert result[0]["metadata"]["llm_task_relevance"]["score_raw"] == 5.0
    assert result[0]["metadata"]["llm_task_relevance"]["score"] == 1.0
    assert result[0]["metadata"]["llm_task_relevance"]["label"] == "relevant"
    assert result[0]["metadata"]["llm_task_relevance"]["task_desc"] == "Math reasoning supervised fine-tuning data."


def test_llm_task_relevance_evaluator_derives_label_when_missing(monkeypatch) -> None:
    def fake_call_llm_json(**_: object) -> tuple[dict[str, object], str]:
        return (
            {
                "score": 2,
                "reason": "Not very relevant.",
            },
            '{"score": 2, "reason": "Not very relevant."}',
        )

    monkeypatch.setattr(
        "dataos_operator.operators.evaluator.llm_task_relevance_evaluator.call_llm_json",
        fake_call_llm_json,
    )

    result = LLMTaskRelevanceEvaluator(
        model="fake-model",
        task_desc="Customer support training data.",
    ).run(
        {"text": "A geometry proof with equations."}
    )

    assert result[0]["metadata"]["llm_task_relevance"]["score"] == 0.25
    assert result[0]["metadata"]["llm_task_relevance"]["label"] == "irrelevant"
