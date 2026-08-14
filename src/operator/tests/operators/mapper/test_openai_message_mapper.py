from __future__ import annotations

from dataos_operator.operators.mapper.openai_message_mapper import OpenAIMessageMapper



def test_openai_message_mapper_maps_qa_record_without_prompt_wrapping() -> None:
    operator = OpenAIMessageMapper(
        source_format="qa",
        query_key="prompt.question",
        answer_key="response.final",
        system_key="system_prompt",
        history_key="turns",
    )

    result = operator.run(
        {
            "id": "sample-qa",
            "prompt": {"question": "解释什么是加速度"},
            "response": {"final": "加速度描述速度变化快慢。"},
            "system_prompt": "你是一位物理老师。",
            "turns": [["什么是速度？", "速度描述位移变化快慢。"]],
            "metadata": {"source": "unit"},
        }
    )

    assert result[0] == {
        "id": "sample-qa",
        "metadata": {"source": "unit"},
        "messages": [
            {"role": "system", "content": "你是一位物理老师。"},
            {"role": "user", "content": "什么是速度？"},
            {"role": "assistant", "content": "速度描述位移变化快慢。"},
            {"role": "user", "content": "解释什么是加速度"},
            {"role": "assistant", "content": "加速度描述速度变化快慢。"},
        ],
    }
    assert result.metadata["kind"] == "mapper"



def test_openai_message_mapper_maps_alpaca_record_without_template_prompt() -> None:
    operator = OpenAIMessageMapper(
        source_format="alpaca",
        instruction_key="prompt.task",
        input_key="prompt.context",
        output_key="response.final",
        system_key="system_prompt",
    )

    result = operator.run(
        {
            "id": "sample-alpaca",
            "prompt": {
                "task": "解释牛顿第二定律",
                "context": "用一句话回答",
            },
            "response": {"final": "牛顿第二定律描述力、质量和加速度之间的关系。"},
            "system_prompt": "你是一位物理老师。",
        }
    )

    assert result[0] == {
        "id": "sample-alpaca",
        "messages": [
            {"role": "system", "content": "你是一位物理老师。"},
            {"role": "user", "content": "解释牛顿第二定律\n用一句话回答"},
            {"role": "assistant", "content": "牛顿第二定律描述力、质量和加速度之间的关系。"},
        ],
    }



def test_openai_message_mapper_omits_empty_alpaca_input_joiner() -> None:
    operator = OpenAIMessageMapper(source_format="alpaca")

    result = operator.run(
        {
            "instruction": "Give three tips for staying healthy.",
            "input": "",
            "output": "Eat well, exercise, and sleep enough.",
        }
    )

    assert result[0]["messages"] == [
        {"role": "user", "content": "Give three tips for staying healthy."},
        {"role": "assistant", "content": "Eat well, exercise, and sleep enough."},
    ]



def test_openai_message_mapper_keeps_existing_messages_shape() -> None:
    operator = OpenAIMessageMapper(source_format="messages")

    result = operator.run(
        {
            "messages": [
                {"role": "system", "content": "你是一个乐于助人的助手。"},
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好，请问有什么可以帮你？"},
            ],
            "metadata": {"source": "chat"},
        }
    )

    assert result[0] == {
        "metadata": {"source": "chat"},
        "messages": [
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好，请问有什么可以帮你？"},
        ],
    }
