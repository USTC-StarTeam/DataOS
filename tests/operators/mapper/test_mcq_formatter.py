from __future__ import annotations

from dataos_operator.operators.mapper.mcq_formatter import McqFormatter


def test_index_answer_becomes_letter_with_lettered_text() -> None:
    operator = McqFormatter()
    record = operator.run(
        [
            {
                "question": "Capital of France?",
                "choices": ["Paris", "London", "Rome", "Berlin"],
                "answer": 0,
            }
        ]
    ).to_records()[0]
    assert record["answer"] == "A"
    assert "A. Paris" in record["output"] and "D. Berlin" in record["output"]
    assert record["output"].startswith("Capital of France?")


def test_answer_given_as_choice_text() -> None:
    operator = McqFormatter()
    record = operator.run(
        [{"question": "q", "choices": ["Paris", "London", "Rome", "Berlin"], "answer": "London"}]
    ).to_records()[0]
    assert record["answer"] == "B"


def test_answer_given_as_lowercase_letter() -> None:
    operator = McqFormatter()
    record = operator.run(
        [{"question": "q", "choices": ["a", "b", "c", "d"], "answer": "c"}]
    ).to_records()[0]
    assert record["answer"] == "C"


def test_custom_letters_field_names_and_dict_choices() -> None:
    operator = McqFormatter(
        question_key="stem",
        choices_key="opts",
        answer_key="gold",
        letters="XYZ",
    )
    record = operator.run(
        [{"stem": "pick", "opts": {"a": "one", "b": "two", "c": "three"}, "gold": 2}]
    ).to_records()[0]
    assert record["gold"] == "Z"
    assert "X. one" in record["output"] and "Z. three" in record["output"]
