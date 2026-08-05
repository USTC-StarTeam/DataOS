"""Extract stable metadata from local image files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dataos_operator.core import EvaluatorOperator, Record
from dataos_operator.core.image import inspect_image, inspect_image_safe, normalize_image_list


class ImageMetadataEvaluator(EvaluatorOperator):
    """Decode local images and attach dimensions, format, size, and integrity metadata.

    Implementation: inspect every reference in ``images_key`` with Pillow and write an ordered
    metadata list to ``output_key``; strict mode raises on unreadable images, while non-strict mode
    records ``readable=False`` plus a diagnostic error for downstream integrity filtering.
    Dependency: requires Pillow; install the project with the ``image`` extra.
    """

    resource_config_keys = ("media_dir",)

    def __init__(
        self,
        *,
        images_key: str = "images",
        output_key: str = "metadata.images",
        media_dir: str | Path | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            images_key=images_key,
            output_key=output_key,
            media_dir=str(media_dir) if media_dir is not None else None,
            strict=strict,
            **kwargs,
        )
        self.images_key = images_key
        self.output_key = output_key
        self.media_dir = media_dir
        self.strict = strict

    def evaluate_record(self, record: Record) -> Record:
        images = normalize_image_list(
            self.get_field(record, self.images_key, None),
            strict=self.strict,
        )
        inspector = inspect_image if self.strict else inspect_image_safe
        metadata = [inspector(image, media_dir=self.media_dir) for image in images]
        return self.set_field(record, self.output_key, metadata)
