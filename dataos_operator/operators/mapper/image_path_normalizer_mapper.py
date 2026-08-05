"""Normalize local and remote image references in a record."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dataos_operator.core import MapperOperator, Record
from dataos_operator.core.image import normalize_image_list, normalize_image_reference


class ImagePathNormalizerMapper(MapperOperator):
    """Normalize top-level image references and optionally validate local file existence.

    Implementation: coerce the configured image field to a list, resolve relative local paths
    against ``media_dir``, optionally make them absolute, preserve supported remote URIs, and
    optionally require every local image to exist.
    Dependency: pure-rule operator, no external dependencies.
    """

    resource_config_keys = ("media_dir",)

    def __init__(
        self,
        *,
        images_key: str = "images",
        media_dir: str | Path | None = None,
        make_absolute: bool = True,
        allow_remote: bool = True,
        check_exists: bool = False,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            images_key=images_key,
            media_dir=str(media_dir) if media_dir is not None else None,
            make_absolute=make_absolute,
            allow_remote=allow_remote,
            check_exists=check_exists,
            strict=strict,
            **kwargs,
        )
        self.images_key = images_key
        self.media_dir = media_dir
        self.make_absolute = make_absolute
        self.allow_remote = allow_remote
        self.check_exists = check_exists
        self.strict = strict

    def map_record(self, record: Record) -> Record:
        images = normalize_image_list(
            self.get_field(record, self.images_key, None),
            strict=self.strict,
        )
        normalized: list[str] = []
        for image in images:
            try:
                reference = normalize_image_reference(
                    image,
                    media_dir=self.media_dir,
                    make_absolute=self.make_absolute,
                    allow_remote=self.allow_remote,
                )
                must_exist = self.check_exists and not _is_remote(reference)
                if must_exist and not Path(reference).is_file():
                    raise FileNotFoundError(f"Image file does not exist: {reference}")
                normalized.append(reference)
            except Exception:
                if self.strict:
                    raise
        return self.set_field(record, self.images_key, normalized)


def _is_remote(reference: str) -> bool:
    return reference.startswith(("http://", "https://", "s3://", "gs://", "oss://"))
