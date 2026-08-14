from __future__ import annotations

from pathlib import Path

import pytest

from dataos_operator.core.image import normalize_image_reference
from dataos_operator.operators.mapper.image_path_normalizer_mapper import (
    ImagePathNormalizerMapper,
)
from dataos_operator.operators.mapper.image_text_message_mapper import ImageTextMessageMapper


def test_image_text_message_mapper_extracts_content_block_images() -> None:
    operator = ImageTextMessageMapper(source_format="messages")

    result = operator.run(
        {
            "id": "sample",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "cat.png"}},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {"role": "assistant", "content": "A cat."},
            ],
        }
    )

    assert result[0] == {
        "id": "sample",
        "images": ["cat.png"],
        "messages": [
            {"role": "user", "content": "<image>\nDescribe this image."},
            {"role": "assistant", "content": "A cat."},
        ],
    }


def test_image_text_message_mapper_maps_fields_and_inserts_placeholders() -> None:
    result = ImageTextMessageMapper(source_format="fields").run(
        {"query": "Compare them.", "answer": "They differ.", "images": ["a.png", "b.png"]}
    )

    assert result[0] == {
        "images": ["a.png", "b.png"],
        "messages": [
            {"role": "user", "content": "<image>\n<image>\nCompare them."},
            {"role": "assistant", "content": "They differ."},
        ],
    }


def test_image_text_message_mapper_rejects_ambiguous_image_order() -> None:
    operator = ImageTextMessageMapper(source_format="messages")

    with pytest.raises(ValueError, match="ordering is ambiguous"):
        operator.run(
            {
                "images": ["top-level.png"],
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "path": "block.png"}],
                    }
                ],
            }
        )


def test_image_path_normalizer_mapper_resolves_relative_paths(tmp_path: Path) -> None:
    image_path = tmp_path / "nested" / "image.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"test")

    result = ImagePathNormalizerMapper(media_dir=tmp_path, check_exists=True).run(
        {"images": ["nested/image.png", "https://example.com/image.png"]}
    )

    assert result[0]["images"] == [str(image_path), "https://example.com/image.png"]


def test_normalize_image_reference_accepts_windows_drive_paths() -> None:
    normalized = normalize_image_reference(r"C:\images\sample.png", make_absolute=False)

    assert normalized == r"C:\images\sample.png"
