from __future__ import annotations

from dataos_operator.operators.selector.code_validity_filter import CodeValidityFilter


def test_keeps_valid_python_and_extracts_from_markdown() -> None:
    operator = CodeValidityFilter()
    result = operator.run(
        [
            {"id": "plain", "output": "def f(x):\n    return x + 1"},
            {"id": "broken", "output": "def f(x)\n    return x +"},
            {"id": "fenced", "output": "```python\ndef g():\n    return 1\n```"},
            {"id": "empty", "output": "   "},
        ]
    ).to_records()
    assert [r["id"] for r in result] == ["plain", "fenced"]


def test_non_python_language_keeps_non_empty() -> None:
    operator = CodeValidityFilter(output_key="code", language="javascript")
    result = operator.run(
        [
            {"code": "function f(){ this is not python }"},
            {"code": ""},
        ]
    ).to_records()
    assert len(result) == 1
