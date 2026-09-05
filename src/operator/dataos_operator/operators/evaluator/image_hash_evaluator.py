"""Compute exact content hashes for local image files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dataos_operator.core import EvaluatorOperator, Record
from dataos_operator.core.image import (
    compute_image_hash,
    normalize_image_list,
    normalize_image_metadata,
    normalize_image_reference,
)


class ImageHashEvaluator(EvaluatorOperator):
    """Compute a streaming cryptographic hash for each local image in a record.

    Implementation: hash files referenced by ``images_key`` and merge the digest into the
    corresponding entry of ``metadata_key`` without discarding metadata produced earlier; in
    non-strict mode, hash failures are stored as diagnostic errors instead of raising.
    Dependency: pure standard-library operator; remote URI downloading is intentionally excluded.
    """

    resource_config_keys = ("media_dir",)

    def __init__(
        self,
        *,
        images_key: str = "images",
        metadata_key: str = "metadata.images",
        algorithm: str = "sha256",
        media_dir: str | Path | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            images_key=images_key,
            metadata_key=metadata_key,
            algorithm=algorithm,
            media_dir=str(media_dir) if media_dir is not None else None,
            strict=strict,
            **kwargs,
        )
        if not algorithm:
            raise ValueError("algorithm must not be empty")
        self.images_key = images_key
        self.metadata_key = metadata_key
        self.algorithm = algorithm.lower()
        self.media_dir = media_dir
        self.strict = strict

    def evaluate_record(self, record: Record) -> Record:
        images = normalize_image_list(
            self.get_field(record, self.images_key, None),
            strict=self.strict,
        )
        existing = normalize_image_metadata(
            self.get_field(record, self.metadata_key, None),
            strict=self.strict,
        )
        metadata: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            item = dict(existing[index]) if index < len(existing) else {}
            try:
                if isinstance(image, str | Path):
                    item.setdefault(
                        "path",
                        normalize_image_reference(image, media_dir=self.media_dir),
                    )
                item[self.algorithm] = compute_image_hash(
                    image,
                    algorithm=self.algorithm,
                    media_dir=self.media_dir,
                )
                item.pop(f"{self.algorithm}_error", None)
            except Exception as exc:  # noqa: BLE001 - non-strict mode records file failures
                if self.strict:
                    raise
                item.setdefault("path", image)
                item[f"{self.algorithm}_error"] = f"{type(exc).__name__}: {exc}"
            metadata.append(item)
        return self.set_field(record, self.metadata_key, metadata)
