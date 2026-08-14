from __future__ import annotations

import pytest

from dataos_operator.operators.mapper.alpaca_mapper import AlpacaMapper


def test_alpaca_mapper_maps_fielded_record_and_deletes_source_keys() -> None:
    operator = AlpacaMapper(
        source_format="fielded",
        instruction_key="prompt.task",
        input_key="prompt.context",
        output_key="response.final",
        system_key="system_prompt",
        history_key="turns",
    )

    result = operator.run(
        {
            "id": "sample-1",
            "prompt": {
                "task": "解释牛顿第一定律",
                "context": "面向高中生",
            },
            "response": {"final": "牛顿第一定律说明物体会保持原有运动状态。"},
            "system_prompt": "你是一位物理老师。",
            "turns": [["什么是力？", "力是物体间的相互作用。"]],
            "metadata": {"source": "unit"},
            "label": "physics",
        }
    )

    assert result[0] == {
        "id": "sample-1",
        "metadata": {"source": "unit"},
        "label": "physics",
        "instruction": "解释牛顿第一定律",
        "input": "面向高中生",
        "output": "牛顿第一定律说明物体会保持原有运动状态。",
        "system": "你是一位物理老师。",
        "history": [["什么是力？", "力是物体间的相互作用。"]],
    }
    assert result.metadata["kind"] == "mapper"


def test_alpaca_mapper_skips_missing_target_fields() -> None:
    operator = AlpacaMapper(source_format="fielded")

    result = operator.run({"id": "sample-2"})

    assert result[0] == {"id": "sample-2"}


def test_alpaca_mapper_can_fill_missing_input_in_fielded_mode() -> None:
    operator = AlpacaMapper(
        source_format="fielded",
        instruction_key="instruction",
        output_key="output",
        always_include_input=True,
    )

    result = operator.run(
        {
            "id": "sample-2b",
            "instruction": "给我一句欢迎语。",
            "output": "欢迎来到 DataOS-Operator。",
        }
    )

    assert result[0] == {
        "id": "sample-2b",
        "instruction": "给我一句欢迎语。",
        "input": "",
        "output": "欢迎来到 DataOS-Operator。",
    }


def test_alpaca_mapper_maps_messages_and_extracts_history() -> None:
    operator = AlpacaMapper(source_format="messages")

    result = operator.run(
        {
            "messages": [
                {"role": "system", "content": "你是一个乐于助人的助手。"},
                {"role": "user", "content": "先解释什么是速度。"},
                {"role": "assistant", "content": "速度描述位移变化快慢。"},
                {"role": "user", "content": "再解释什么是加速度。"},
                {"role": "assistant", "content": "加速度描述速度变化快慢。"},
            ],
            "metadata": {"source": "chat"},
        }
    )

    assert result[0]["instruction"] == "再解释什么是加速度。"
    assert result[0]["output"] == "加速度描述速度变化快慢。"
    assert result[0]["system"] == "你是一个乐于助人的助手。"
    assert result[0]["history"] == [["先解释什么是速度。", "速度描述位移变化快慢。"]]
    assert "input" not in result[0]
    assert "messages" not in result[0]


def test_alpaca_mapper_can_always_include_empty_input_from_messages() -> None:
    operator = AlpacaMapper(source_format="messages", always_include_input=True)

    result = operator.run(
        {
            "messages": [
                {"role": "user", "content": "给一个 Python 例子。"},
                {"role": "assistant", "content": "print(\"hello\")"},
            ]
        }
    )

    assert result[0] == {
        "instruction": "给一个 Python 例子。",
        "input": "",
        "output": 'print("hello")',
    }


def test_alpaca_mapper_skips_missing_message_fields() -> None:
    operator = AlpacaMapper(source_format="messages")

    result = operator.run({"id": "sample-3"})

    assert result[0] == {"id": "sample-3"}


def test_alpaca_mapper_rejects_tool_roles_in_message_mode() -> None:
    operator = AlpacaMapper(source_format="messages")

    with pytest.raises(ValueError, match="unsupported role for Alpaca mapping"):
        operator.run(
            {
                "messages": [
                    {"role": "user", "content": "查询上海天气"},
                    {"role": "tool", "content": "晴，28 度"},
                    {"role": "assistant", "content": "上海今天晴，28 度。"},
                ]
            }
        )
