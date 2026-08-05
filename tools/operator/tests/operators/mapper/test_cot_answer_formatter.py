from __future__ import annotations

from dataos_operator.operators.mapper.cot_answer_formatter import CotAnswerFormatter


def _output(record: dict) -> str:
    return record["output"]


def test_extracts_from_boxed_source() -> None:
    operator = CotAnswerFormatter(reasoning_key="solution")
    record = operator.run(
        [{"question": "q", "solution": "We reason and get \\boxed{42} here."}]
    ).to_records()[0]
    assert _output(record).endswith("#### 42")


def test_extracts_from_hashed_source_and_dedupes_marker() -> None:
    operator = CotAnswerFormatter(reasoning_key="solution")
    record = operator.run(
        [{"question": "q", "solution": "Step one.\nStep two.\n#### 18"}]
    ).to_records()[0]
    out = _output(record)
    # the pre-existing "#### 18" is stripped and re-appended exactly once.
    assert out == "Step one.\nStep two.\n#### 18"
    assert out.count("####") == 1


def test_extracts_from_trailing_number_source() -> None:
    operator = CotAnswerFormatter(reasoning_key="solution")
    record = operator.run([{"question": "q", "solution": "The total is 7"}]).to_records()[0]
    assert _output(record) == "The total is 7\n#### 7"


def test_uses_explicit_answer_key_when_present() -> None:
    operator = CotAnswerFormatter(reasoning_key="solution", answer_key="gold")
    record = operator.run(
        [{"solution": "Some reasoning without a marker.", "gold": "99"}]
    ).to_records()[0]
    assert _output(record).endswith("#### 99")
    assert record["gold"] == "99"


def test_custom_template() -> None:
    operator = CotAnswerFormatter(
        reasoning_key="solution", answer_template="The answer is {answer}."
    )
    record = operator.run([{"solution": "so \\boxed{5}"}]).to_records()[0]
    assert _output(record).endswith("The answer is 5.")
