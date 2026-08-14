"""按输入顺序直接拼接所有 dataset 的 mixer。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Dataset, MixerOperator, Record


class ConcatMixer(MixerOperator):
    """Concatenate one or more datasets in input order into a single dataset.

    Implementation: flatten the records of all input datasets in order, without changing
    content or aligning fields.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def mix(self, data: Sequence[Dataset]) -> Sequence[Record]:
        return [record for dataset in data for record in dataset]
