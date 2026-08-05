"""Normalize field-based or content-block image-text samples."""

from __future__ import annotations

from typing import Any

from dataos_operator.core import MapperOperator, Record
from dataos_operator.core.image import IMAGE_PLACEHOLDER, normalize_image_list
from dataos_operator.core.image_text_formats import (
    add_missing_image_placeholders,
    normalize_image_text_messages,
)


class ImageTextMessageMapper(MapperOperator):
    """Map image-text records into normalized ``messages`` plus top-level ``images``.

    Implementation: accept OpenAI-style string/content-block messages or field-based query/answer
    records, extract image references in encounter order, and emit string messages containing one
    ``<image>`` placeholder per image while preserving unrelated record fields.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        *,
        source_format: str = "auto",
        messages_key: str = "messages",
        images_key: str = "images",
        query_key: str = "query",
        answer_key: str = "answer",
        placeholder: str = IMAGE_PLACEHOLDER,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_format=source_format,
            messages_key=messages_key,
            images_key=images_key,
            query_key=query_key,
            answer_key=answer_key,
            placeholder=placeholder,
            strict=strict,
            **kwargs,
        )
        if source_format not in {"auto", "messages", "fields"}:
            raise ValueError("source_format must be one of: auto, messages, fields")
        if not placeholder:
            raise ValueError("placeholder must not be empty")
        self.source_format = source_format
        self.messages_key = messages_key
        self.images_key = images_key
        self.query_key = query_key
        self.answer_key = answer_key
        self.placeholder = placeholder
        self.strict = strict

    def map_record(self, record: Record) -> Record:
        source_format = self._resolved_source_format(record)
        if source_format == "messages":
            return self._map_messages(record)
        return self._map_fields(record)

    def _resolved_source_format(self, record: Record) -> str:
        if self.source_format != "auto":
            return self.source_format
        has_messages = self.get_field(record, self.messages_key, None) is not None
        return "messages" if has_messages else "fields"

    def _map_messages(self, record: Record) -> Record:
        raw_messages = self.require_field(record, self.messages_key)
        messages, extracted_images = normalize_image_text_messages(
            raw_messages,
            placeholder=self.placeholder,
            strict=self.strict,
        )
        existing_images = normalize_image_list(
            self.get_field(record, self.images_key, None),
            strict=self.strict,
        )
        if extracted_images and existing_images:
            if self.strict:
                raise ValueError(
                    "record contains both content-block images and a top-level images field; "
                    "their ordering is ambiguous"
                )
            images = extracted_images
        else:
            images = extracted_images or existing_images
        updated = self.set_field(record, self.messages_key, messages)
        return self.set_field(updated, self.images_key, images)

    def _map_fields(self, record: Record) -> Record:
        query = str(self.require_field(record, self.query_key))
        answer = str(self.require_field(record, self.answer_key))
        images = normalize_image_list(
            self.get_field(record, self.images_key, None),
            strict=self.strict,
        )
        query = add_missing_image_placeholders(
            query,
            image_count=len(images),
            placeholder=self.placeholder,
            strict=self.strict,
        )
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        updated = dict(record)
        for key in {self.query_key, self.answer_key}:
            if "." not in key:
                updated.pop(key, None)
        updated = dict(self.set_field(updated, self.messages_key, messages))
        return self.set_field(updated, self.images_key, images)
