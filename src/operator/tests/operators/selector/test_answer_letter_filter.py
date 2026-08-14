from __future__ import annotations

from dataos_operator.operators.selector.answer_letter_filter import AnswerLetterFilter


def test_keeps_records_with_explicit_letter_answer() -> None:
    operator = AnswerLetterFilter()
    result = operator.run(
        [
            {"id": "phrase", "output": "After analysis, the answer is C."},
            {"id": "paren", "output": "I pick (B) here."},
            {"id": "single", "output": "A"},
            {"id": "hashed", "output": "reasoning\n#### D"},
            {"id": "prose", "output": "A cat sat on a mat and did nothing."},
        ]
    ).to_records()
    assert [r["id"] for r in result] == ["phrase", "paren", "single", "hashed"]


def test_custom_letters_and_key() -> None:
    operator = AnswerLetterFilter(output_key="resp", letters="EFGH")
    result = operator.run(
        [
            {"resp": "The answer is F"},
            {"resp": "the answer is A"},  # A not in EFGH
        ]
    ).to_records()
    assert len(result) == 1 and result[0]["resp"] == "The answer is F"
