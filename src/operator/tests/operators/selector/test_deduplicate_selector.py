from __future__ import annotations

import pytest

from dataos_operator.operators.selector.deduplicate_selector import DeduplicateSelector


def test_deduplicate_selector_keeps_first_record_by_default() -> None:
    operator = DeduplicateSelector(key="text")

    result = operator.run(
        [
            {"id": "a", "text": "Hello"},
            {"id": "b", "text": "hello"},
            {"id": "c", "text": "world"},
        ]
    )

    assert result.to_records() == [
        {"id": "a", "text": "Hello"},
        {"id": "c", "text": "world"},
    ]
    assert result.metadata["kind"] == "selector"
    assert result.metadata["input_records"] == 3
    assert result.metadata["output_records"] == 2


def test_deduplicate_selector_can_keep_last_record() -> None:
    operator = DeduplicateSelector(key="text", keep="last")

    result = operator.run(
        [
            {"id": "a", "text": "Alpha"},
            {"id": "b", "text": "beta"},
            {"id": "c", "text": "alpha"},
        ]
    )

    assert result.to_records() == [
        {"id": "b", "text": "beta"},
        {"id": "c", "text": "alpha"},
    ]


def test_deduplicate_selector_can_use_case_sensitive_mode() -> None:
    operator = DeduplicateSelector(key="text", case_sensitive=True)

    result = operator.run(
        [
            {"id": "a", "text": "Hello"},
            {"id": "b", "text": "hello"},
        ]
    )

    assert result.to_records() == [
        {"id": "a", "text": "Hello"},
        {"id": "b", "text": "hello"},
    ]


def test_deduplicate_selector_accepts_nested_keys_and_multiple_datasets() -> None:
    operator = DeduplicateSelector(key="payload.text")

    result = operator.run(
        [
            [{"id": "a", "payload": {"text": "one"}}],
            [
                {"id": "b", "payload": {"text": "ONE"}},
                {"id": "c", "payload": {"text": "two"}},
            ],
        ]
    )

    assert result.to_records() == [
        {"id": "a", "payload": {"text": "one"}},
        {"id": "c", "payload": {"text": "two"}},
    ]
    assert result.metadata["input_items"] == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"key": ""}, "key must be a non-empty string"),
        ({"keep": "middle"}, "keep must be one of: first, last"),
    ],
)
def test_deduplicate_selector_validates_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DeduplicateSelector(**kwargs)
