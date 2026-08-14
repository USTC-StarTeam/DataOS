from __future__ import annotations

import json

from dataos_operator.operators.selector.function_call_filter import FunctionCallFilter


def test_keeps_only_valid_call_lists() -> None:
    operator = FunctionCallFilter()
    result = operator.run(
        [
            {"output": json.dumps([{"name": "f", "arguments": {"x": 1}}])},
            {"output": "I do not have access to functions."},
            {"output": "<functioncall> {name: f}"},
            {"output": json.dumps([{"name": "g", "arguments": {}}])},
        ]
    ).to_records()
    assert [json.loads(r["output"])[0]["name"] for r in result] == ["f", "g"]


def test_custom_key_and_multiple_datasets() -> None:
    operator = FunctionCallFilter(output_key="label")
    result = operator.run(
        [
            [{"label": json.dumps([{"name": "a", "arguments": {}}])}, {"label": "nope"}],
            [{"label": 123}],
        ]
    ).to_records()
    assert len(result) == 1
