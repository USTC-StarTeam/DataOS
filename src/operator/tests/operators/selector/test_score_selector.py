from __future__ import annotations

import pytest

from dataos_operator.operators.selector.score_selector import ScoreSelector


def test_score_selector_filters_by_score_range() -> None:
    operator = ScoreSelector(score_key="metadata.perplexity", min_score=2.0, max_score=4.0)

    result = operator.run(
        [
            {"id": "a", "metadata": {"perplexity": 1.5}},
            {"id": "b", "metadata": {"perplexity": 2.5}},
            {"id": "c", "metadata": {"perplexity": 4.0}},
            {"id": "d", "metadata": {"perplexity": 4.5}},
        ]
    )

    assert result.to_records() == [
        {"id": "b", "metadata": {"perplexity": 2.5}},
        {"id": "c", "metadata": {"perplexity": 4.0}},
    ]


def test_score_selector_accepts_multiple_datasets_and_numeric_strings() -> None:
    operator = ScoreSelector(score_key="score", max_score=3.0)

    result = operator.run(
        [
            [{"id": "a", "score": "2.5"}],
            [{"id": "b", "score": 3.5}, {"id": "c", "score": 1}],
        ]
    )

    assert result.to_records() == [{"id": "a", "score": "2.5"}, {"id": "c", "score": 1}]


def test_score_selector_can_keep_missing_scores() -> None:
    operator = ScoreSelector(score_key="score", min_score=1.0, keep_missing=True)

    result = operator.run([{"id": "a"}, {"id": "b", "score": 2.0}])

    assert result.to_records() == [{"id": "a"}, {"id": "b", "score": 2.0}]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"score_key": "score"}, "cannot both be None"),
        ({"score_key": "score", "min_score": 3.0, "max_score": 2.0}, "greater than max_score"),
    ],
)
def test_score_selector_validates_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ScoreSelector(**kwargs)
