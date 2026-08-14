from __future__ import annotations

import pytest

from dataos_operator.operators.selector.random_sample_selector import RandomSampleSelector


def test_random_sample_selector_selects_requested_number_of_records(monkeypatch) -> None:
    operator = RandomSampleSelector(sample_size=2)

    monkeypatch.setattr(
        "dataos_operator.operators.selector.random_sample_selector.random.sample",
        lambda records, sample_size: list(records[:sample_size]),
    )

    result = operator.run(
        [
            {"id": "a", "text": "alpha"},
            {"id": "b", "text": "beta"},
            {"id": "c", "text": "gamma"},
        ]
    )

    assert result.to_records() == [
        {"id": "a", "text": "alpha"},
        {"id": "b", "text": "beta"},
    ]
    assert result.metadata["kind"] == "selector"
    assert result.metadata["input_records"] == 3
    assert result.metadata["output_records"] == 2


def test_random_sample_selector_accepts_multiple_datasets(monkeypatch) -> None:
    operator = RandomSampleSelector(sample_size=3)

    monkeypatch.setattr(
        "dataos_operator.operators.selector.random_sample_selector.random.sample",
        lambda records, sample_size: list(records[-sample_size:]),
    )

    result = operator.run(
        [
            [{"id": "a"}, {"id": "b"}],
            [{"id": "c"}, {"id": "d"}],
        ]
    )

    assert result.to_records() == [{"id": "b"}, {"id": "c"}, {"id": "d"}]
    assert result.metadata["input_items"] == 2
    assert result.metadata["output_records"] == 3


def test_random_sample_selector_can_return_all_records_when_not_strict() -> None:
    operator = RandomSampleSelector(sample_size=5, strict=False)

    result = operator.run(
        [
            {"id": "a", "text": "alpha"},
            {"id": "b", "text": "beta"},
        ]
    )

    assert result.to_records() == [
        {"id": "a", "text": "alpha"},
        {"id": "b", "text": "beta"},
    ]


def test_random_sample_selector_rejects_insufficient_records_in_strict_mode() -> None:
    operator = RandomSampleSelector(sample_size=3, strict=True)

    with pytest.raises(ValueError, match="exceeds available records"):
        operator.run([{"id": "a"}, {"id": "b"}])


def test_random_sample_selector_validates_configuration() -> None:
    with pytest.raises(ValueError, match="sample_size must be > 0"):
        RandomSampleSelector(sample_size=0)
