from __future__ import annotations

from dataos_operator.operators.mapper import PairedConversationMapper


def test_paired_conversation_mapper_preserves_original_multimodal_fields() -> None:
    record = {
        "id": "x",
        "images": {"bytes": b"image", "path": None},
        "texts": [
            {"user": "Question one?", "assistant": "Answer one."},
            {"user": "Question two?", "assistant": "Answer two."},
        ],
    }

    result = PairedConversationMapper().run(record)[0]

    assert result["images"] == record["images"]
    assert result["texts"] == record["texts"]
    assert result["messages"] == [
        {"role": "user", "content": "Question one?"},
        {"role": "assistant", "content": "Answer one."},
        {"role": "user", "content": "Question two?"},
        {"role": "assistant", "content": "Answer two."},
    ]
