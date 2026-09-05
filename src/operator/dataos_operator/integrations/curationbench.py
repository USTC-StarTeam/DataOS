"""Format-preserving helpers for Curation-Bench datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from dataos_operator.view import DatasetView


def load_pool(path: str | Path) -> DatasetView:
    """Open the official pool lazily; image payloads remain in Arrow."""

    return DatasetView.from_disk(path)


def with_reference_indices(
    pool: DatasetView,
    source_indices: Iterable[int],
    *,
    output_key: str = "metadata.reference",
) -> DatasetView:
    """Mark source rows belonging to a prior selection manifest."""

    reference = set(source_indices)
    return pool.with_columns({output_key: [index in reference for index in pool.source_indices]})


def save_submission(
    selection: DatasetView,
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    """Save original rows and optionally persist reusable source-row identities."""

    output = selection.save_to_disk(output_path)
    if manifest_path is not None:
        manifest = Path(manifest_path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", encoding="utf-8") as stream:
            for order, source_index in enumerate(selection.source_indices):
                stream.write(
                    json.dumps(
                        {"order": order, "source_index": source_index},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return output


def load_manifest_indices(path: str | Path) -> list[int]:
    """Read source indices written by :func:`save_submission`."""

    indices: list[int] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row: dict[str, Any] = json.loads(line)
        value = row.get("source_index")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("manifest source_index must be an integer")
        indices.append(value)
    return indices


def selected_source_indices(selection: DatasetView) -> Sequence[int]:
    """Expose the immutable source identities for reporting and assertions."""

    return selection.source_indices
