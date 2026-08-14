from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from dataos_operator.operators.evaluator.image_hash_evaluator import ImageHashEvaluator
from dataos_operator.operators.evaluator.image_metadata_evaluator import ImageMetadataEvaluator


def _write_image(path: Path, *, size: tuple[int, int] = (8, 4)) -> None:
    Image.new("RGB", size, color=(10, 20, 30)).save(path)


def test_image_metadata_evaluator_extracts_stable_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    _write_image(image_path)

    result = ImageMetadataEvaluator().run({"images": [str(image_path)]})
    metadata = result[0]["metadata"]["images"][0]

    assert metadata["path"] == str(image_path)
    assert metadata["readable"] is True
    assert metadata["format"] == "PNG"
    assert metadata["width"] == 8
    assert metadata["height"] == 4
    assert metadata["pixels"] == 32
    assert metadata["aspect_ratio"] == 2.0
    assert metadata["bytes"] == image_path.stat().st_size


def test_image_metadata_evaluator_non_strict_mode_records_failure(tmp_path: Path) -> None:
    broken_path = tmp_path / "broken.png"
    broken_path.write_text("not an image", encoding="utf-8")

    result = ImageMetadataEvaluator(strict=False).run({"images": [str(broken_path)]})
    metadata = result[0]["metadata"]["images"][0]

    assert metadata["exists"] is True
    assert metadata["readable"] is False
    assert "error" in metadata


def test_image_hash_evaluator_merges_hash_into_existing_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    _write_image(image_path)
    expected = hashlib.sha256(image_path.read_bytes()).hexdigest()

    result = ImageHashEvaluator().run(
        {
            "images": [str(image_path)],
            "metadata": {"images": [{"width": 8, "height": 4}]},
        }
    )

    metadata = result[0]["metadata"]["images"][0]
    assert metadata["width"] == 8
    assert metadata["height"] == 4
    assert metadata["sha256"] == expected


def test_image_hash_evaluator_non_strict_mode_records_invalid_uri() -> None:
    result = ImageHashEvaluator(strict=False).run({"images": ["ftp://example.com/image.png"]})
    metadata = result[0]["metadata"]["images"][0]

    assert metadata["path"] == "ftp://example.com/image.png"
    assert "unsupported image URI scheme" in metadata["sha256_error"]
