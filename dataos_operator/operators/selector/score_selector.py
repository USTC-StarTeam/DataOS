"""按指定分数字段范围筛选样本的 selector。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Dataset, Record, SelectorOperator


class ScoreSelector(SelectorOperator):
    """Filter samples by the numeric range of a score field.

    Implementation: read ``score_key`` (dotted paths supported, numeric strings auto-cast to
    float) and keep records whose score lies in ``[min_score, max_score]``; missing scores
    are handled per ``keep_missing``.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        score_key: str,
        min_score: float | None = None,
        max_score: float | None = None,
        *,
        keep_missing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            score_key=score_key,
            min_score=min_score,
            max_score=max_score,
            keep_missing=keep_missing,
            **kwargs,
        )
        if min_score is None and max_score is None:
            raise ValueError("min_score and max_score cannot both be None")
        if min_score is not None and max_score is not None and min_score > max_score:
            raise ValueError("min_score cannot be greater than max_score")

        self.score_key = score_key
        self.min_score = min_score
        self.max_score = max_score
        self.keep_missing = keep_missing

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        selected: list[Record] = []
        for dataset in data:
            for record in dataset:
                score_value = self.get_field(record, self.score_key, None)
                if score_value is None:
                    if self.keep_missing:
                        selected.append(record)
                    continue

                score = _coerce_numeric(score_value, self.score_key)
                if self.min_score is not None and score < self.min_score:
                    continue
                if self.max_score is not None and score > self.max_score:
                    continue
                selected.append(record)
        return selected


def _coerce_numeric(value: Any, score_key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Score field {score_key!r} must be numeric, got boolean")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Score field {score_key!r} is not numeric: {value!r}") from exc
    raise ValueError(f"Score field {score_key!r} is not numeric: {value!r}")
