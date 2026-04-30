"""Source operator that materializes a Hugging Face DatasetArtifact from raw input.

Supported sources (selected via ``spec['source']``):

- ``parquet``: one or more local parquet files to load via ``datasets``.
- ``hf_hub``: a HuggingFace Hub repo id (snapshot or stream) loaded via ``load_dataset``.
- ``hf_local``: a local directory already in HF ``save_to_disk`` format.

Common spec fields:

- ``output_path``: absolute directory the materialized DatasetArtifact will live at.
- ``rename`` (optional): mapping of source column → target column.
- ``keep_columns`` (optional): list of columns to retain (others are dropped).
- ``required_columns`` (optional): list of columns that must be non-empty per row.
- ``train_test_split`` (optional): ``{"test_size": 0.02, "seed": 42}`` to create a test split.
- ``max_train_samples`` / ``max_test_samples`` (optional): cap each split (deterministic).
- ``name`` (optional): artifact display name.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox


def _coerce_path_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(f"Expected str or list[str] for parquet path; got {type(value).__name__}")


def _load_parquet(spec: dict[str, Any]) -> Dataset | DatasetDict:
    paths = _coerce_path_list(spec.get("path"))
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"parquet source not found: {path}")
    return load_dataset("parquet", data_files=paths, split="train")


def _load_hf_hub(spec: dict[str, Any]) -> Dataset | DatasetDict:
    repo_id = spec["repo_id"]
    config = spec.get("config")
    split = spec.get("split")
    cache_dir = spec.get("cache_dir")
    return load_dataset(repo_id, name=config, split=split, cache_dir=cache_dir)


def _load_hf_local(spec: dict[str, Any]) -> Dataset | DatasetDict:
    path = spec["path"]
    if not Path(path).exists():
        raise FileNotFoundError(f"hf_local source not found: {path}")
    obj = load_from_disk(path)
    if not isinstance(obj, (Dataset, DatasetDict)):
        raise ValueError(f"hf_local at {path} is not a Dataset or DatasetDict")
    return obj


_LOADERS = {
    "parquet": _load_parquet,
    "hf_hub": _load_hf_hub,
    "hf_local": _load_hf_local,
}


def _apply_rename(dataset: Dataset, rename: dict[str, str]) -> Dataset:
    actual_rename = {src: dst for src, dst in rename.items() if src in dataset.column_names and src != dst}
    if not actual_rename:
        return dataset
    return dataset.rename_columns(actual_rename)


def _apply_keep_columns(dataset: Dataset, keep_columns: list[str]) -> Dataset:
    drop = [col for col in dataset.column_names if col not in keep_columns]
    if not drop:
        return dataset
    return dataset.remove_columns(drop)


def _apply_required_columns(dataset: Dataset, required_columns: list[str]) -> Dataset:
    missing = [col for col in required_columns if col not in dataset.column_names]
    if missing:
        raise ValueError(f"required_columns not present in source: {missing}")

    def _is_nonempty(example: dict[str, Any]) -> bool:
        for col in required_columns:
            value = example.get(col)
            if value is None:
                return False
            if isinstance(value, str) and not value.strip():
                return False
        return True

    return dataset.filter(_is_nonempty)


def _maybe_split(dataset: Dataset, split_cfg: dict[str, Any] | None) -> DatasetDict:
    if split_cfg is None:
        return DatasetDict({"train": dataset})
    test_size = float(split_cfg.get("test_size", 0.02))
    seed = int(split_cfg.get("seed", 42))
    if test_size <= 0:
        return DatasetDict({"train": dataset})
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({"train": splits["train"], "test": splits["test"]})


def _cap_split(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None or dataset.num_rows <= max_samples:
        return dataset
    return dataset.select(range(max_samples))


def _resolve_dataset(spec: dict[str, Any]) -> Dataset | DatasetDict:
    source = spec.get("source")
    if source not in _LOADERS:
        raise ValueError(
            f"Unsupported import source '{source}'. Supported: {sorted(_LOADERS)}"
        )
    return _LOADERS[source](spec)


def _ensure_dataset_dict(obj: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(obj, DatasetDict):
        return obj
    return DatasetDict({"train": obj})


class ImportDatasetOperator(Operator):
    name = "import_dataset"
    description = "Materialize a Hugging Face DatasetArtifact from a raw source spec."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        del inputs
        spec = dict(kwargs)
        loaded = _resolve_dataset(spec)
        dataset_dict = _ensure_dataset_dict(loaded)

        rename = spec.get("rename") or {}
        keep_columns = spec.get("keep_columns")
        required_columns = spec.get("required_columns") or []
        max_train = spec.get("max_train_samples")
        max_test = spec.get("max_test_samples")
        split_cfg = spec.get("train_test_split")

        processed: dict[str, Dataset] = {}
        for split, dataset in dataset_dict.items():
            if rename:
                dataset = _apply_rename(dataset, rename)
            if keep_columns:
                dataset = _apply_keep_columns(dataset, keep_columns)
            if required_columns:
                dataset = _apply_required_columns(dataset, required_columns)
            processed[split] = dataset

        if "test" not in processed and split_cfg is not None and "train" in processed:
            split_dd = _maybe_split(processed["train"], split_cfg)
            processed = {**processed, **{k: v for k, v in split_dd.items()}}

        if "train" in processed and max_train is not None:
            processed["train"] = _cap_split(processed["train"], int(max_train))
        if "test" in processed and max_test is not None:
            processed["test"] = _cap_split(processed["test"], int(max_test))

        final = DatasetDict(processed)

        artifact_id = make_artifact_id("dataset")
        artifact_name = spec.get("name") or f"imported-{spec.get('source')}"
        output_path = spec.get("output_path")
        target_path = (
            Path(output_path)
            if output_path
            else sandbox.artifact_dir("dataset", artifact_id)
        )
        if target_path.exists():
            shutil.rmtree(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = DatasetArtifact(
            artifact_id=artifact_id,
            name=artifact_name,
            path=target_path,
            parent_ids=[],
            producer=self.name,
            meta={
                "import_spec": {k: v for k, v in spec.items() if k != "output_path"},
                "splits_after_import": list(final.keys()),
            },
        )
        artifact.save(final)
        sandbox.log(
            "import_dataset",
            f"source={spec.get('source')}\nname={artifact_name}\npath={target_path}\n"
            f"splits={list(final.keys())}\nrows={artifact.meta.get('num_rows')}\n",
        )
        return artifact
