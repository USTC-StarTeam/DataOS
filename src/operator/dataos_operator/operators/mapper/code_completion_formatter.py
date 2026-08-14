"""把 (prompt, code) 代码样本规范成干净可执行的函数补全文本。"""

from __future__ import annotations

import re
from typing import Any

from dataos_operator.core import MapperOperator, Record

_FENCE = re.compile(r"^\s*```[^\n]*\n?|\n?```\s*$")


class CodeCompletionFormatter(MapperOperator):
    """Format a (prompt, code) pair into one clean, executable function-completion text.

    Implementation: read ``prompt_key`` (typically a signature + docstring) and ``code_key``
    (a body or a full function), stripping any Markdown code fences from the code. If the code
    already defines a function it is used as-is; otherwise the prompt and code are concatenated so
    the completion stays a single runnable function. Field names are parameterized and the
    signature/body detection is content-based (``def`` presence), so no dataset layout is assumed.
    Dependency: pure-rule operator, no external dependencies.
    """

    operator_name = "code_completion_formatter"

    def __init__(
        self,
        prompt_key: str = "prompt",
        code_key: str = "canonical_solution",
        *,
        output_key: str = "output",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            prompt_key=prompt_key,
            code_key=code_key,
            output_key=output_key,
            **kwargs,
        )
        self.prompt_key = prompt_key
        self.code_key = code_key
        self.output_key = output_key

    def map_record(self, record: Record) -> Record:
        prompt = _as_text(self.get_field(record, self.prompt_key, ""))
        code = _strip_fences(_as_text(self.get_field(record, self.code_key, "")))

        if "def " in code:
            output = code
        elif prompt:
            joiner = "" if (not prompt or prompt.endswith("\n")) else "\n"
            output = f"{prompt}{joiner}{code}"
        else:
            output = code

        result = dict(record)
        result[self.output_key] = output
        return result


def _strip_fences(text: str) -> str:
    if "```" not in text:
        return text
    return _FENCE.sub("", text.strip())


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)
