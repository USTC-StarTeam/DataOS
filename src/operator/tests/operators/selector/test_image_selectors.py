from __future__ import annotations

from PIL import Image

from dataos_operator import Dataset
from dataos_operator.operators.selector.image_aspect_ratio_filter import (
    ImageAspectRatioFilter,
)
from dataos_operator.operators.selector.image_count_filter import ImageCountFilter
from dataos_operator.operators.selector.image_exact_deduplicate_selector import (
    ImageExactDeduplicateSelector,
)
from dataos_operator.operators.selector.image_format_filter import ImageFormatFilter
from dataos_operator.operators.selector.image_integrity_filter import ImageIntegrityFilter
from dataos_operator.operators.selector.image_shape_filter import ImageShapeFilter
from dataos_operator.operators.selector.image_size_filter import ImageSizeFilter
from dataos_operator.operators.selector.image_text_alignment_filter import (
    ImageTextAlignmentFilter,
)


def _ids(result: Dataset) -> list[str]:
    return [record["id"] for record in result]


def test_image_integrity_filter_drops_unreadable_images() -> None:
    records = [
        {"id": "good", "metadata": {"images": [{"exists": True, "readable": True}]}},
        {"id": "bad", "metadata": {"images": [{"exists": True, "readable": False}]}},
        {"id": "missing"},
    ]

    assert _ids(ImageIntegrityFilter().run(records)) == ["good"]


def test_image_text_alignment_filter_matches_placeholder_count() -> None:
    records = [
        {
            "id": "aligned",
            "images": ["a.png", "b.png"],
            "messages": [{"role": "user", "content": "<image> then <image>"}],
        },
        {
            "id": "mismatch",
            "images": ["a.png", "b.png"],
            "messages": [{"role": "user", "content": "<image>"}],
        },
    ]

    assert _ids(ImageTextAlignmentFilter().run(records)) == ["aligned"]


def test_image_count_filter_applies_inclusive_bounds() -> None:
    records = [
        {"id": "one", "images": ["a.png"]},
        {"id": "two", "images": ["a.png", "b.png"]},
        {"id": "three", "images": ["a.png", "b.png", "c.png"]},
    ]

    assert _ids(ImageCountFilter(min_count=1, max_count=2).run(records)) == ["one", "two"]


def test_image_count_filter_accepts_a_scalar_pil_image() -> None:
    records = [{"id": "pil", "images": Image.new("RGB", (2, 2))}]

    assert _ids(ImageCountFilter(min_count=1, max_count=1).run(records)) == ["pil"]


def test_image_shape_filter_supports_all_and_any_strategies() -> None:
    records = [
        {
            "id": "mixed",
            "metadata": {
                "images": [
                    {"width": 1024, "height": 768, "pixels": 786432},
                    {"width": 64, "height": 64, "pixels": 4096},
                ]
            },
        }
    ]

    assert _ids(ImageShapeFilter(min_width=512, strategy="all").run(records)) == []
    assert _ids(ImageShapeFilter(min_width=512, strategy="any").run(records)) == ["mixed"]


def test_image_ratio_size_and_format_filters() -> None:
    records = [
        {
            "id": "accepted",
            "metadata": {
                "images": [
                    {"aspect_ratio": 1.5, "bytes": 1500, "format": "JPG"},
                ]
            },
        },
        {
            "id": "rejected",
            "metadata": {
                "images": [
                    {"aspect_ratio": 4.0, "bytes": 50, "format": "GIF"},
                ]
            },
        },
    ]

    ratio_result = ImageAspectRatioFilter(min_ratio=1.0, max_ratio=2.0).run(records)
    size_result = ImageSizeFilter(min_bytes=100, max_bytes=2000).run(records)
    format_result = ImageFormatFilter(allowed_formats=["jpeg", "png"]).run(records)

    assert _ids(ratio_result) == ["accepted"]
    assert _ids(size_result) == ["accepted"]
    assert _ids(format_result) == ["accepted"]


def test_image_exact_deduplicate_selector_supports_order_insensitive_sets() -> None:
    records = [
        {"id": "first", "metadata": {"images": [{"sha256": "a"}, {"sha256": "b"}]}},
        {"id": "same-order", "metadata": {"images": [{"sha256": "a"}, {"sha256": "b"}]}},
        {"id": "reverse", "metadata": {"images": [{"sha256": "b"}, {"sha256": "a"}]}},
        {"id": "missing"},
    ]

    ordered = ImageExactDeduplicateSelector().run(records)
    unordered = ImageExactDeduplicateSelector(order_sensitive=False, keep_missing=True).run(records)

    assert _ids(ordered) == ["first", "reverse"]
    assert _ids(unordered) == ["first", "missing"]
