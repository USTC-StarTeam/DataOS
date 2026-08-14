"""Normalization helpers for image-text chat messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dataos_operator.core.image import IMAGE_PLACEHOLDER

_ROLE_ALIASES = {
    "human": "user",
    "gpt": "assistant",
    "function_call": "assistant",
    "observation": "tool",
}
_SUPPORTED_ROLES = {"system", "user", "assistant", "tool"}
_TEXT_BLOCK_TYPES = {"text", "input_text", "output_text"}
_IMAGE_BLOCK_TYPES = {"image", "image_url", "input_image"}


def normalize_image_text_messages(
    value: Any,
    *,
    placeholder: str = IMAGE_PLACEHOLDER,
    strict: bool = True,
) -> tuple[list[dict[str, str]], list[str]]:
    """Normalize string or content-block messages and extract ordered image references."""

    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError("messages must be a sequence of message mappings")

    messages: list[dict[str, str]] = []
    images: list[str] = []
    for item in value:
        if not isinstance(item, Mapping):
            if strict:
                raise ValueError("each message must be a mapping")
            continue
        role_value = item.get("role", item.get("from"))
        if role_value is None:
            if strict:
                raise ValueError("message role is missing")
            continue
        role = _ROLE_ALIASES.get(str(role_value).strip().lower(), str(role_value).strip().lower())
        if role not in _SUPPORTED_ROLES:
            if strict:
                raise ValueError(f"unsupported image-text message role: {role_value}")
            continue

        content = item.get("content", item.get("value"))
        if isinstance(content, str):
            normalized_content = content
        elif isinstance(content, Sequence) and not isinstance(content, bytes | bytearray):
            normalized_content, extracted = _normalize_content_blocks(
                content,
                placeholder=placeholder,
                strict=strict,
            )
            images.extend(extracted)
        elif content is None:
            if strict:
                raise ValueError("message content is missing")
            normalized_content = ""
        else:
            if strict:
                raise ValueError("message content must be a string or content-block sequence")
            normalized_content = str(content)
        messages.append({"role": role, "content": normalized_content})
    return messages, images


def add_missing_image_placeholders(
    text: str,
    *,
    image_count: int,
    placeholder: str = IMAGE_PLACEHOLDER,
    strict: bool = True,
) -> str:
    """Ensure a field-based query contains one placeholder per image."""

    existing = text.count(placeholder)
    if existing == image_count:
        return text
    if existing > 0:
        if strict:
            raise ValueError(
                f"query contains {existing} image placeholders but record has {image_count} images"
            )
        return text
    if image_count == 0:
        return text
    prefix = "\n".join(placeholder for _ in range(image_count))
    return f"{prefix}\n{text}" if text else prefix


def _normalize_content_blocks(
    blocks: Sequence[Any],
    *,
    placeholder: str,
    strict: bool,
) -> tuple[str, list[str]]:
    parts: list[str] = []
    images: list[str] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            if strict:
                raise ValueError("message content blocks must be mappings")
            continue
        block_type = str(block.get("type", "")).strip().lower()
        if block_type in _TEXT_BLOCK_TYPES:
            text = block.get("text", block.get("content"))
            if text is None:
                if strict:
                    raise ValueError("text content block is missing text")
                continue
            parts.append(str(text))
            continue
        if block_type in _IMAGE_BLOCK_TYPES:
            reference = _image_reference_from_block(block)
            if reference is None:
                if strict:
                    raise ValueError("image content block is missing an image reference")
                continue
            parts.append(placeholder)
            images.append(reference)
            continue
        if strict:
            raise ValueError(f"unsupported message content block type: {block_type!r}")
    return "\n".join(part for part in parts if part != ""), images


def _image_reference_from_block(block: Mapping[str, Any]) -> str | None:
    value = block.get("image_url", block.get("image", block.get("url", block.get("path"))))
    if isinstance(value, Mapping):
        value = value.get("url", value.get("path"))
    if value is None:
        return None
    text = str(value).strip()
    return text or None
