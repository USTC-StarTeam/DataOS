from __future__ import annotations

from py_agent.providers.message_transform import transform_chat_messages
from py_agent.providers.openai_responses import OpenAICompatibleProvider


def test_transform_keeps_valid_tool_pairing() -> None:
    msgs = [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "write", "arguments": "{\"path\":\"a.txt\"}"},
                }
            ],
        },
        {"role": "tool", "name": "write", "tool_call_id": "t1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    out = transform_chat_messages(msgs)
    assert out == msgs


def test_transform_drops_orphan_tool_message() -> None:
    msgs = [
        {"role": "user", "content": "u1"},
        {"role": "tool", "name": "write", "tool_call_id": "t1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    out = transform_chat_messages(msgs)
    assert not any(m.get("role") == "tool" for m in out)


def test_transform_inserts_missing_tool_result() -> None:
    msgs = [
        {"role": "user", "content": "u1"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\":\"pwd\"}"},
                }
            ],
        },
        {"role": "user", "content": "u2"},
    ]
    out = transform_chat_messages(msgs)
    assert out[2]["role"] == "tool"
    assert out[2]["tool_call_id"] == "t1"
    assert "missing tool result" in out[2]["content"]


def test_provider_uses_transformed_messages() -> None:
    seen = {}

    def fake_transport(url: str, payload: dict, headers: dict) -> dict:
        seen["payload"] = payload
        return {"choices": [{"message": {"content": "ok"}}]}

    provider = OpenAICompatibleProvider(
        base_url="https://example.com",
        api_key="token",
        transport=fake_transport,
    )
    provider.chat(
        model="glm-5",
        tools=[],
        messages=[
            {"role": "user", "content": "u1"},
            {"role": "tool", "name": "write", "tool_call_id": "t1", "content": "orphan"},
        ],
    )
    out_messages = seen["payload"]["messages"]
    assert len(out_messages) == 1
    assert out_messages[0]["role"] == "user"
