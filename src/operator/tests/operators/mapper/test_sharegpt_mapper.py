from __future__ import annotations

import pytest

from dataos_operator.operators.mapper.sharegpt_mapper import ShareGPTMapper


def test_sharegpt_mapper_maps_fielded_record_and_deletes_source_keys() -> None:
    operator = ShareGPTMapper(
        source_format="fielded",
        instruction_key="prompt.task",
        input_key="prompt.context",
        output_key="response.final",
        system_key="system_prompt",
        history_key="turns",
        tools_key="tool_spec",
        user_joiner="\n\n",
    )

    result = operator.run(
        {
            "id": "sample-1",
            "prompt": {
                "task": "解释牛顿第二定律",
                "context": "用一句话回答",
            },
            "response": {"final": "牛顿第二定律描述力、质量和加速度之间的关系。"},
            "system_prompt": "你是一位物理老师。",
            "turns": [["什么是惯性？", "惯性是物体保持原有状态的性质。"]],
            "tool_spec": "[{\"name\": \"search\"}]",
            "metadata": {"source": "unit"},
        }
    )

    assert result[0] == {
        "id": "sample-1",
        "metadata": {"source": "unit"},
        "conversations": [
            {"from": "human", "value": "什么是惯性？"},
            {"from": "gpt", "value": "惯性是物体保持原有状态的性质。"},
            {"from": "human", "value": "解释牛顿第二定律\n\n用一句话回答"},
            {"from": "gpt", "value": "牛顿第二定律描述力、质量和加速度之间的关系。"},
        ],
        "system": "你是一位物理老师。",
        "tools": "[{\"name\": \"search\"}]",
    }
    assert result.metadata["kind"] == "mapper"


def test_sharegpt_mapper_skips_missing_target_fields() -> None:
    operator = ShareGPTMapper(source_format="fielded")

    result = operator.run({"id": "sample-2"})

    assert result[0] == {"id": "sample-2"}


def test_sharegpt_mapper_maps_messages_and_extracts_system() -> None:
    operator = ShareGPTMapper(source_format="messages")

    result = operator.run(
        {
            "messages": [
                {"role": "system", "content": "你是一个乐于助人的助手。"},
                {"role": "user", "content": "请介绍动量守恒。"},
                {"role": "assistant", "content": "动量守恒表示系统总动量保持不变。"},
                {"role": "tool", "content": "检索到教材章节 2.3"},
                {"role": "assistant", "content": "教材中常见表述是封闭系统总动量不变。"},
            ],
            "metadata": {"source": "chat"},
        }
    )

    assert result[0]["system"] == "你是一个乐于助人的助手。"
    assert result[0]["conversations"] == [
        {"from": "human", "value": "请介绍动量守恒。"},
        {"from": "gpt", "value": "动量守恒表示系统总动量保持不变。"},
        {"from": "observation", "value": "检索到教材章节 2.3"},
        {"from": "gpt", "value": "教材中常见表述是封闭系统总动量不变。"},
    ]
    assert "tools" not in result[0]
    assert "messages" not in result[0]


def test_sharegpt_mapper_skips_missing_message_fields() -> None:
    operator = ShareGPTMapper(source_format="messages")

    result = operator.run({"id": "sample-3"})

    assert result[0] == {"id": "sample-3"}


def test_sharegpt_mapper_validates_role_order() -> None:
    operator = ShareGPTMapper(source_format="messages")

    with pytest.raises(ValueError, match="invalid ShareGPT role order"):
        operator.run(
            {
                "messages": [
                    {"role": "assistant", "content": "我先说。"},
                    {"role": "user", "content": "这不合法。"},
                ]
            }
        )
