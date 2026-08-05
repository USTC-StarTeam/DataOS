from __future__ import annotations

import pytest

from dataos_operator.operators.selector.length_filter import LengthFilter


def test_length_filter_selects_records_within_range() -> None:
    operator = LengthFilter(
        text_key="payload.text",
        min_length=3,
        max_length=5,
    )

    result = operator.run(
        [
            {"payload": {"text": "hi"}},
            {"payload": {"text": "hello"}},
            {"payload": {"text": "welcome"}},
        ]
    )

    assert result.to_records() == [{"payload": {"text": "hello"}}]
    assert result.metadata["kind"] == "selector"
    assert result.metadata["input_records"] == 3
    assert result.metadata["output_records"] == 1


def test_length_filter_accepts_multiple_datasets() -> None:
    operator = LengthFilter(text_key="text", min_length=2, max_length=4)

    result = operator.run(
        [
            [{"text": "a"}, {"text": "good"}],
            [{"text": "tool"}, {"text": "welcome"}],
        ]
    )

    assert result.to_records() == [
        {"text": "good"},
        {"text": "tool"},
    ]
    assert result.metadata["input_items"] == 2
    assert result.metadata["output_records"] == 2


def test_length_filter_can_keep_missing_records() -> None:
    operator = LengthFilter(text_key="text", min_length=2, keep_missing=True)

    result = operator.run(
        [
            {"id": "missing"},
            {"id": "short", "text": "a"},
            {"id": "long-enough", "text": "abc"},
        ]
    )

    assert result.to_records() == [
        {"id": "missing"},
        {"id": "long-enough", "text": "abc"},
    ]


def test_length_filter_can_strip_whitespace_before_counting() -> None:
    operator = LengthFilter(text_key="text", min_length=2, max_length=4, strip_whitespace=True)

    result = operator.run([
        {"text": "  a  "},
        {"text": "  abc  "},
    ])

    assert result.to_records() == [{"text": "  abc  "}]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "At least one of min_length or max_length must be set"),
        ({"min_length": -1}, "min_length must be >= 0"),
        ({"max_length": -1}, "max_length must be >= 0"),
        ({"min_length": 5, "max_length": 3}, "min_length must be <= max_length"),
        ({"min_length": 1, "length_unit": "token"}, "Unsupported length_unit"),
    ],
)
def test_length_filter_validates_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LengthFilter(**kwargs)
