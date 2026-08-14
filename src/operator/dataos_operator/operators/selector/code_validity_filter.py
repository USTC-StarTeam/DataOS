"""只保留 output 是合法(可解析)代码的样本。"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator

_MISSING = object()
_CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    match = _CODE_BLOCK.search(text)
    return match.group(1) if match else text


class CodeValidityFilter(SelectorOperator):
    """Keep only rows whose ``output`` is valid (parseable) code.

    Implementation: read the ``output_key`` text, pull the first fenced code block when present
    (otherwise use the whole string), and for ``language == "python"`` keep the row only when
    ``ast.parse`` succeeds; for other languages the row is kept when the extracted code is
    non-empty (no parser available). The field name and language are parameterized and the check is
    content-based, so no dataset layout is assumed. Use it to drop syntactically broken targets
    before training on a code benchmark (e.g. HumanEval) that executes the completion.
    Dependency: pure-rule operator, no external dependencies (uses the stdlib ``ast``).
    """

    operator_name = "code_validity_filter"

    def __init__(
        self,
        *,
        output_key: str = "output",
        language: str = "python",
        **kwargs: Any,
    ) -> None:
        super().__init__(output_key=output_key, language=language, **kwargs)
        self.output_key = output_key
        self.language = language

    def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
        selected: list[Record] = []
        for dataset in data:
            for record in dataset:
                value = self.get_field(record, self.output_key, _MISSING)
                if value is _MISSING or value is None:
                    continue
                if self._is_valid(str(value)):
                    selected.append(record)
        return selected

    def _is_valid(self, text: str) -> bool:
        code = _extract_code(text).strip()
        if not code:
            return False
        if self.language.lower() != "python":
            return True
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        return True
