"""随机选择固定数量 record 的 selector。"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Dataset, Record, SelectorOperator


class RandomSampleSelector(SelectorOperator):
    """Randomly draw a fixed number of records from one or more datasets.

    Implementation: merge all inputs and draw ``sample_size`` rows via ``random.sample``;
    when there are too few rows, either raise or return all per ``strict``. A given ``seed``
    uses a dedicated ``random.Random`` instance for reproducibility; otherwise the global
    random source is used.
    Dependency: pure-rule operator, no external dependencies.

    Reproducibility: without ``seed`` each run selects different rows, so the output dataset
    fingerprint drifts and replay of a declarative recipe (fingerprint consistency check) is
    bound to fail; always pass ``seed`` for recipes / reproducible runs.
    """

    def __init__(
        self, sample_size: int, *, strict: bool = True, seed: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(sample_size=sample_size, strict=strict, seed=seed, **kwargs)
        if sample_size <= 0:
            raise ValueError("sample_size must be > 0")

        self.sample_size = sample_size
        self.strict = strict
        self.seed = seed

    def select(self, data: Sequence[Dataset]) -> Sequence[Record]:
        records = [record for dataset in data for record in dataset]
        if len(records) < self.sample_size:
            if self.strict:
                raise ValueError(
                    f"sample_size {self.sample_size} exceeds available records {len(records)}"
                )
            return records
        rng = random.Random(self.seed) if self.seed is not None else random
        return rng.sample(records, self.sample_size)
