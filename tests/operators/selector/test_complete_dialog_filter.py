from __future__ import annotations

from dataos_operator.operators.selector.complete_dialog_filter import CompleteDialogFilter


def test_keeps_dialogues_with_nonempty_assistant_reply() -> None:
    operator = CompleteDialogFilter()
    result = operator.run(
        [
            {"id": "ok", "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there"},
            ]},
            {"id": "empty-assistant", "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "   "},
            ]},
            {"id": "no-assistant", "messages": [{"role": "user", "content": "hi"}]},
        ]
    ).to_records()
    assert [r["id"] for r in result] == ["ok"]


def test_sharegpt_style_via_custom_keys() -> None:
    operator = CompleteDialogFilter(
        messages_key="conversations", role_key="from", content_key="value"
    )
    result = operator.run(
        [
            {"id": "gpt", "conversations": [
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "an actual answer"},
            ]},
            {"id": "human-only", "conversations": [{"from": "human", "value": "q"}]},
        ]
    ).to_records()
    assert [r["id"] for r in result] == ["gpt"]


def test_min_assistant_chars_threshold() -> None:
    operator = CompleteDialogFilter(min_assistant_chars=10)
    result = operator.run(
        [
            {"id": "short", "messages": [{"role": "assistant", "content": "hi"}]},
            {"id": "long", "messages": [{"role": "assistant", "content": "a long enough reply"}]},
        ]
    ).to_records()
    assert [r["id"] for r in result] == ["long"]
