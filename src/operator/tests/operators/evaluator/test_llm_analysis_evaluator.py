from __future__ import annotations

from dataos_operator.operators.evaluator.llm_analysis_evaluator import LLMAnalysisEvaluator


def test_llm_analysis_evaluator_maps_messages_and_normalizes_score(monkeypatch) -> None:
    def fake_call_llm_json(**_: object) -> tuple[dict[str, object], str]:
        return (
            {
                "dimension_scores": {
                    "quality": 4,
                    "relevance": 5,
                    "informativeness": 4,
                    "clarity": 3,
                    "safety": 5,
                },
                "label": "keep",
                "reason": "Strong sample.",
            },
            '{"dimension_scores": {"quality": 4, "relevance": 5, "informativeness": 4, "clarity": 3, "safety": 5}, "label": "keep", "reason": "Strong sample."}',
        )

    monkeypatch.setattr(
        "dataos_operator.operators.evaluator.llm_analysis_evaluator.call_llm_json",
        fake_call_llm_json,
    )

    result = LLMAnalysisEvaluator(model="fake-model").run(
        {
            "messages": [
                {"role": "user", "content": "Explain Newton's first law."},
                {"role": "assistant", "content": "It describes inertia."},
            ]
        }
    )

    assert result[0]["metadata"]["llm_analysis"]["dimension_scores"]["quality"] == 4.0
    assert result[0]["metadata"]["llm_analysis"]["score_raw"] == 4.2
    assert result[0]["metadata"]["llm_analysis"]["score"] == 0.8
    assert result[0]["metadata"]["llm_analysis"]["label"] == "keep"
    assert result[0]["metadata"]["llm_analysis"]["reason"] == "Strong sample."


def test_llm_analysis_evaluator_uses_field_input_when_messages_missing(monkeypatch) -> None:
    def fake_call_llm_json(**_: object) -> tuple[dict[str, object], str]:
        return (
            {
                "dimension_scores": {
                    "quality": 2,
                    "relevance": 2,
                    "informativeness": 2,
                    "clarity": 2,
                    "safety": 4,
                },
                "reason": "Weak sample.",
            },
            '{"dimension_scores": {"quality": 2, "relevance": 2, "informativeness": 2, "clarity": 2, "safety": 4}, "reason": "Weak sample."}',
        )

    monkeypatch.setattr(
        "dataos_operator.operators.evaluator.llm_analysis_evaluator.call_llm_json",
        fake_call_llm_json,
    )

    result = LLMAnalysisEvaluator(model="fake-model").run(
        {
            "query": "Write a poem.",
            "answer": "hello",
        }
    )

    assert result[0]["metadata"]["llm_analysis"]["score"] == 0.35
    assert result[0]["metadata"]["llm_analysis"]["label"] == "drop"
