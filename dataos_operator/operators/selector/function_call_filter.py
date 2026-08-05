"""只保留 output 能解析成合法函数调用列表的样本。"""

from __future__ import annotations

import json
from collections.abc import Sequence

from dataos_operator.core.contracts import Dataset, SelectorOperator


class FunctionCallFilter(SelectorOperator):
    """Keep only rows whose ``output`` parses as a valid function-call list.

    Implementation: ``json.loads`` of the ``output_key`` value must be a list of
    ``{"name", "arguments"}`` dicts; everything else (natural-language answers, "no access to
    functions" negatives, loose ``<functioncall>`` shapes) is dropped. The field name is
    parameterized and the check is content-based, so it works on any tool-calling source. Use it
    before training on a function-calling benchmark (e.g. BFCL) whose eval parses exact call
    syntax — a plain map -> sample pipeline over a dirty tool-call source otherwise trains on
    garbage and scores near zero.
    Dependency: pure-rule operator, no external dependencies.
    """

    operator_name = "function_call_filter"

    def __init__(self, *, output_key: str = "output", **kwargs: object) -> None:
        super().__init__(output_key=output_key, **kwargs)
        self.output_key = output_key

    def select(self, data: Sequence[Dataset]) -> Dataset:
        rows = [record for dataset in data for record in dataset]
        return [r for r in rows if self._is_valid_call(self.get_field(r, self.output_key, ""))]

    @staticmethod
    def _is_valid_call(output: object) -> bool:
        if not isinstance(output, str):
            return False
        try:
            calls = json.loads(output)
        except (ValueError, TypeError):
            return False
        return isinstance(calls, list) and all(
            isinstance(c, dict) and "name" in c and "arguments" in c for c in calls
        )
