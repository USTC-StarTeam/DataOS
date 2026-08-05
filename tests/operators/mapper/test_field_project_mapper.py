from __future__ import annotations

import pytest

from dataos_operator.operators.mapper.field_project_mapper import FieldProjectMapper


def test_field_project_mapper_keeps_selected_fields_only() -> None:
    operator = FieldProjectMapper(keep_fields=("instruction", "input", "output", "metadata"))

    result = operator.run(
        {
            "instruction": "解释牛顿第二定律。",
            "input": "面向初中生",
            "output": "力等于质量乘加速度。",
            "metadata": {"source": "unit", "score": 0.9},
            "text": "drop me",
        }
    )

    assert result[0] == {
        "instruction": "解释牛顿第二定律。",
        "input": "面向初中生",
        "output": "力等于质量乘加速度。",
        "metadata": {"source": "unit", "score": 0.9},
    }
    assert result.metadata["operator"] == "field_project_mapper"


def test_field_project_mapper_skips_missing_fields_when_not_strict() -> None:
    operator = FieldProjectMapper(
        keep_fields=("instruction", "input", "output", "metadata"),
        strict=False,
    )

    result = operator.run(
        {
            "instruction": "给一个问候语。",
            "output": "你好。",
        }
    )

    assert result[0] == {
        "instruction": "给一个问候语。",
        "output": "你好。",
    }


def test_field_project_mapper_rejects_missing_fields_when_strict() -> None:
    operator = FieldProjectMapper(keep_fields=("instruction", "input"))

    with pytest.raises(KeyError, match="Required field is missing: input"):
        operator.run({"instruction": "only instruction"})
