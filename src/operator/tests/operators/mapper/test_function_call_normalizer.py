from __future__ import annotations

import json

from dataos_operator.operators.mapper.function_call_normalizer import FunctionCallNormalizer


def _messages(record: dict) -> list[dict]:
    return record["messages"]


def test_xlam_style_parameters_dict_normalized() -> None:
    operator = FunctionCallNormalizer()
    record = operator.run(
        [
            {
                "query": "What is the weather in NYC?",
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Look up weather",
                        "parameters": {
                            "location": {"type": "str", "description": "city"},
                            "unit": {"type": "str", "default": "c"},
                        },
                    }
                ],
                "answer": [{"name": "get_weather", "arguments": {"location": "NYC"}}],
            }
        ]
    ).to_records()[0]

    tools = record["tools"]
    schema = tools[0]["parameters"]
    assert schema["type"] == "object"
    assert schema["properties"]["location"]["type"] == "string"
    # unit has a default -> not required; location has none -> required.
    assert schema["required"] == ["location"]

    messages = _messages(record)
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "What is the weather in NYC?"}
    assistant = messages[2]["content"]
    assert "<tool_call>" in assistant and "</tool_call>" in assistant
    call = json.loads(assistant.split("<tool_call>")[1].split("</tool_call>")[0])
    assert call == {"name": "get_weather", "arguments": {"location": "NYC"}}


def test_glaive_style_tagged_call_and_string_args_normalized() -> None:
    operator = FunctionCallNormalizer()
    record = operator.run(
        [
            {
                "query": "weather?",
                "tools": json.dumps(
                    [
                        {
                            "name": "get_weather",
                            "description": "..",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        }
                    ]
                ),
                # single-quoted arguments blob inside a <functioncall> tag.
                "answer": "<functioncall> {\"name\": \"get_weather\", "
                "\"arguments\": '{\"location\": \"NYC\"}'}",
            }
        ]
    ).to_records()[0]

    assert record["tools"][0]["parameters"]["properties"]["location"]["type"] == "string"
    assistant = _messages(record)[2]["content"]
    call = json.loads(assistant.split("<tool_call>")[1].split("</tool_call>")[0])
    assert call["name"] == "get_weather"
    assert call["arguments"] == {"location": "NYC"}


def test_custom_field_names_tag_and_type_map() -> None:
    operator = FunctionCallNormalizer(
        tools_key="functions",
        call_key="label",
        query_key="prompt",
        tool_call_tag="call",
        type_map={"integer": "number"},
    )
    record = operator.run(
        [
            {
                "prompt": "add",
                "functions": [{"name": "add", "parameters": {"a": {"type": "integer"}}}],
                "label": [{"name": "add", "arguments": {"a": 1}}],
            }
        ]
    ).to_records()[0]

    assert record["functions"][0]["parameters"]["properties"]["a"]["type"] == "number"
    assistant = _messages(record)[2]["content"]
    assert assistant.startswith("<call>")
