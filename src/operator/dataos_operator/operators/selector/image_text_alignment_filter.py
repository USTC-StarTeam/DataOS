"""Filter records by image placeholder alignment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dataos_operator.core import Record, SelectorOperator
from dataos_operator.core.image import (
    IMAGE_PLACEHOLDER,
    count_image_placeholders,
    normalize_image_list,
)


class ImageTextAlignmentFilter(SelectorOperator):
    """Keep records with one textual placeholder for every top-level image reference.

    Implementation: count ``placeholder`` occurrences across normalized messages and require the
    count to equal the number of entries in ``images_key`` while also satisfying optional image
    count bounds; this catches mismatched image/message ordering before model training.
    Dependency: pure-rule operator, no external dependencies.
    """

    def __init__(
        self,
        *,
        messages_key: str = "messages",
        images_key: str = "images",
        placeholder: str = IMAGE_PLACEHOLDER,
        min_images: int = 1,
        max_images: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            messages_key=messages_key,
            images_key=images_key,
            placeholder=placeholder,
            min_images=min_images,
            max_images=max_images,
            **kwargs,
        )
        if not placeholder:
            raise ValueError("placeholder must not be empty")
        if min_images < 0:
            raise ValueError("min_images must be >= 0")
        if max_images is not None and max_images < min_images:
            raise ValueError("max_images must be >= min_images")
        self.messages_key = messages_key
        self.images_key = images_key
        self.placeholder = placeholder
        self.min_images = min_images
        self.max_images = max_images

    def select(self, data: Sequence[Sequence[Record]]) -> Sequence[Record]:
        selected: list[Record] = []
        for dataset in data:
            for record in dataset:
                images = normalize_image_list(
                    self.get_field(record, self.images_key, None),
                    strict=False,
                )
                count = len(images)
                if count < self.min_images:
                    continue
                if self.max_images is not None and count > self.max_images:
                    continue
                messages = self.get_field(record, self.messages_key, None)
                if count_image_placeholders(messages, placeholder=self.placeholder) == count:
                    selected.append(record)
        return selected
