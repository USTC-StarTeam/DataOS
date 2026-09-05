"""Index-preserving view over a large Hugging Face dataset.

``Dataset`` remains the record-oriented container used by the text operators.  This view is the
bounded-memory companion for curation tasks: operators keep source row indices and lightweight
derived columns, while images stay in the original Arrow dataset until final export.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, overload

from dataos_operator.core.contracts import Record, get_field, set_field


@dataclass(frozen=True)
class ViewBatch:
    """One bounded batch from a :class:`DatasetView`."""

    positions: tuple[int, ...]
    source_indices: tuple[int, ...]
    columns: Mapping[str, tuple[Any, ...]]

    def __getitem__(self, key: str) -> tuple[Any, ...]:
        return self.columns[key]


class DatasetView(Sequence[Record]):
    """A lazy selection of source rows plus lightweight, index-aligned derived columns."""

    _dataos_dataset_view = True

    def __init__(
        self,
        source: Any,
        *,
        indices: Sequence[int] | None = None,
        derived_columns: Mapping[str, Sequence[Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not hasattr(source, "__len__") or not hasattr(source, "__getitem__"):
            raise TypeError("source must provide __len__ and __getitem__")
        resolved_indices: Sequence[int] = range(len(source)) if indices is None else tuple(indices)
        for index in resolved_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("indices must contain integers")
            if index < 0 or index >= len(source):
                raise IndexError(f"source index out of range: {index}")
        if len(set(resolved_indices)) != len(resolved_indices):
            raise ValueError("indices must be unique")

        columns = {key: tuple(values) for key, values in (derived_columns or {}).items()}
        for key, values in columns.items():
            if len(values) != len(resolved_indices):
                raise ValueError(
                    f"derived column {key!r} has {len(values)} rows; "
                    f"expected {len(resolved_indices)}"
                )

        self._source = source
        self._indices = resolved_indices
        self._derived_columns = columns
        self._metadata = dict(metadata or {})

    @classmethod
    def from_disk(cls, path: str | Path) -> DatasetView:
        """Open a Hugging Face ``save_to_disk`` dataset without materializing its rows."""

        import datasets as hfds

        source = hfds.load_from_disk(str(path))
        if isinstance(source, hfds.DatasetDict):
            raise TypeError("DatasetView requires a Dataset, not a DatasetDict")
        return cls(source, metadata={"loader": "hf_disk", "path": str(path)})

    @property
    def source(self) -> Any:
        return self._source

    @property
    def source_indices(self) -> Sequence[int]:
        return self._indices

    @property
    def derived_columns(self) -> Mapping[str, Sequence[Any]]:
        return MappingProxyType(self._derived_columns)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(self._metadata)

    def __len__(self) -> int:
        return len(self._indices)

    @overload
    def __getitem__(self, index: int) -> Record: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Record]: ...

    def __getitem__(self, index: int | slice) -> Record | Sequence[Record]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        position = index + len(self) if index < 0 else index
        if position < 0 or position >= len(self):
            raise IndexError(index)
        record: Record = dict(self._source[self._indices[position]])
        for key, values in self._derived_columns.items():
            record = set_field(record, key, values[position])
        return record

    def __iter__(self) -> Iterator[Record]:
        for position in range(len(self)):
            yield self[position]

    def with_columns(self, columns: Mapping[str, Sequence[Any]]) -> DatasetView:
        """Return a view with added or replaced lightweight derived columns."""

        merged = dict(self._derived_columns)
        for key, values in columns.items():
            materialized = tuple(values)
            if len(materialized) != len(self):
                raise ValueError(
                    f"derived column {key!r} has {len(materialized)} rows; expected {len(self)}"
                )
            merged[key] = materialized
        return DatasetView(
            self._source,
            indices=self._indices,
            derived_columns=merged,
            metadata=self._metadata,
        )

    def with_metadata(self, metadata: Mapping[str, Any]) -> DatasetView:
        return DatasetView(
            self._source,
            indices=self._indices,
            derived_columns=self._derived_columns,
            metadata=metadata,
        )

    def select_positions(self, positions: Sequence[int]) -> DatasetView:
        """Select positions in this view while preserving their original source identities."""

        normalized = tuple(positions)
        if len(set(normalized)) != len(normalized):
            raise ValueError("positions must be unique")
        for position in normalized:
            if isinstance(position, bool) or not isinstance(position, int):
                raise TypeError("positions must contain integers")
            if position < 0 or position >= len(self):
                raise IndexError(f"view position out of range: {position}")
        columns = {
            key: tuple(values[position] for position in normalized)
            for key, values in self._derived_columns.items()
        }
        return DatasetView(
            self._source,
            indices=tuple(self._indices[position] for position in normalized),
            derived_columns=columns,
            metadata=self._metadata,
        )

    def iter_batches(
        self,
        columns: Sequence[str],
        *,
        batch_size: int = 4096,
    ) -> Iterator[ViewBatch]:
        """Yield requested raw/derived columns without touching unrelated image columns."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        requested = tuple(dict.fromkeys(columns))
        raw_columns = [key for key in requested if key not in self._derived_columns]
        raw_roots = tuple(dict.fromkeys(key.split(".", 1)[0] for key in raw_columns))

        for start in range(0, len(self), batch_size):
            stop = min(start + batch_size, len(self))
            positions = tuple(range(start, stop))
            source_indices = tuple(self._indices[position] for position in positions)
            raw_batch = self._read_source_batch(source_indices, raw_roots)
            values: dict[str, tuple[Any, ...]] = {}
            for key in requested:
                derived = self._derived_columns.get(key)
                if derived is not None:
                    values[key] = tuple(derived[position] for position in positions)
                    continue
                root = key.split(".", 1)[0]
                root_values = raw_batch[root]
                if key == root:
                    values[key] = tuple(root_values)
                else:
                    values[key] = tuple(
                        get_field({root: value}, key, None) for value in root_values
                    )
            yield ViewBatch(
                positions=positions,
                source_indices=source_indices,
                columns=MappingProxyType(values),
            )

    def save_to_disk(self, path: str | Path) -> Path:
        """Export only original source rows, preserving the source dataset's features."""

        if not hasattr(self._source, "select") or not hasattr(self._source, "save_to_disk"):
            raise TypeError("source does not support Hugging Face select/save_to_disk")
        selected = self._source.select(list(self._indices))
        selected.save_to_disk(str(path))
        return Path(path)

    def _read_source_batch(
        self,
        source_indices: Sequence[int],
        roots: Sequence[str],
    ) -> Mapping[str, Sequence[Any]]:
        if not roots:
            return {}
        if hasattr(self._source, "select") and hasattr(self._source, "select_columns"):
            selected = self._source.select(list(source_indices)).select_columns(list(roots))
            return selected[:]
        rows = [self._source[index] for index in source_indices]
        return {root: [record[root] for record in rows] for root in roots}
