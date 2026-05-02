"""Exact deduplication on a configurable composite key.

Spec fields (all optional unless noted):

- ``keys`` (required): list of column names. The dedup key per row is the joined
  string of those columns; the first occurrence within each split wins.
- ``normalize`` (default ``True``): strip + lowercase string values before hashing.
- ``per_split`` (default ``True``): when ``False``, dedup across all splits using
  the same key set (still preserves the first occurrence in iteration order over
  splits in their declared order).
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox


def _find_latest_dataset(inputs: list[Artifact]) -> DatasetArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, DatasetArtifact):
            return artifact
    raise ValueError("dedup requires at least one DatasetArtifact input")


def _normalize_value(value: Any, normalize: bool) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().lower() if normalize else value
    return repr(value)


def _row_key(row: dict[str, Any], keys: list[str], normalize: bool) -> str:
    return "\x1f".join(_normalize_value(row.get(key), normalize) for key in keys)


def _dedup_split(
    dataset: Dataset,
    keys: list[str],
    normalize: bool,
    seen: set[str],
) -> tuple[Dataset, int]:
    missing = [key for key in keys if key not in dataset.column_names]
    if missing:
        raise ValueError(f"dedup keys not present in dataset: {missing}")
    keep_indices: list[int] = []
    for index, row in enumerate(dataset):
        key = _row_key(row, keys, normalize)
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(index)
    deduped = dataset.select(keep_indices)
    return deduped, dataset.num_rows - deduped.num_rows


class DedupOperator(Operator):
    name = "dedup"
    description = "Exact deduplication on a composite key over one or more columns."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = _find_latest_dataset(inputs)
        keys = list(kwargs.get("keys") or [])
        if not keys:
            raise ValueError("dedup requires non-empty 'keys'")
        normalize = bool(kwargs.get("normalize", True))
        per_split = bool(kwargs.get("per_split", True))

        dataset_obj = source.load()
        removed_counts: dict[str, int] = {}
        if isinstance(dataset_obj, DatasetDict):
            shared: set[str] = set()
            deduped_splits: dict[str, Dataset] = {}
            for split, split_dataset in dataset_obj.items():
                seen = set() if per_split else shared
                deduped_splits[split], removed_counts[split] = _dedup_split(
                    split_dataset, keys, normalize, seen
                )
            deduped: Dataset | DatasetDict = DatasetDict(deduped_splits)
        else:
            seen: set[str] = set()
            deduped, removed = _dedup_split(dataset_obj, keys, normalize, seen)
            removed_counts = {"train": removed}

        artifact_id = make_artifact_id("dataset")
        target = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-dedup",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "dedup_keys": keys,
                "dedup_normalize": normalize,
                "dedup_per_split": per_split,
                "dedup_removed_counts": removed_counts,
            },
        )
        target.save(deduped)
        sandbox.log(
            "dedup",
            f"source={source.artifact_id}\noutput={target.artifact_id}\n"
            f"keys={keys}\nremoved={removed_counts}\n",
        )
        return target
