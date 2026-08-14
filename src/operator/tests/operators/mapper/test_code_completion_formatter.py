from __future__ import annotations

import ast

from dataos_operator.operators.mapper.code_completion_formatter import CodeCompletionFormatter


def test_prompt_plus_body_becomes_runnable_function() -> None:
    operator = CodeCompletionFormatter(prompt_key="prompt", code_key="canonical_solution")
    record = operator.run(
        [
            {
                "prompt": 'def add(a, b):\n    """Add."""\n',
                "canonical_solution": "    return a + b\n",
            }
        ]
    ).to_records()[0]
    ast.parse(record["output"])  # runnable
    assert "def add" in record["output"] and "return a + b" in record["output"]


def test_full_function_in_markdown_fence_is_extracted() -> None:
    operator = CodeCompletionFormatter(prompt_key="prompt", code_key="code")
    record = operator.run(
        [{"prompt": "", "code": "```python\ndef sub(a, b):\n    return a - b\n```"}]
    ).to_records()[0]
    assert record["output"] == "def sub(a, b):\n    return a - b"
    ast.parse(record["output"])


def test_custom_field_names() -> None:
    operator = CodeCompletionFormatter(prompt_key="stub", code_key="solution")
    record = operator.run(
        [{"stub": "def f(x):\n", "solution": "    return x * 2\n"}]
    ).to_records()[0]
    ast.parse(record["output"])
