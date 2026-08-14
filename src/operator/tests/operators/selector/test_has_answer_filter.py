from __future__ import annotations

from dataos_operator.operators.selector.has_answer_filter import HasAnswerFilter


def test_keeps_records_across_answer_formats() -> None:
    operator = HasAnswerFilter()
    result = operator.run(
        [
            {"id": "boxed", "output": "reason \\boxed{42}"},
            {"id": "hashed", "output": "reason\n#### 18"},
            {"id": "trailing", "output": "the total is 7"},
            {"id": "prose", "output": "I am not sure about this one."},
            {"id": "empty", "output": ""},
        ]
    ).to_records()
    assert [r["id"] for r in result] == ["boxed", "hashed", "trailing"]


def test_custom_key_and_pattern() -> None:
    operator = HasAnswerFilter(output_key="resp", answer_pattern=r"ANSWER:\s*(\w+)")
    result = operator.run(
        [
            {"resp": "ANSWER: dog"},
            {"resp": "no marker here"},
        ]
    ).to_records()
    assert len(result) == 1 and result[0]["resp"] == "ANSWER: dog"
