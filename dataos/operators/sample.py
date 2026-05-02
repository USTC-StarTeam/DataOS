"""Deterministic subsampling for a Hugging Face DatasetArtifact.

Spec fields (all optional unless noted):

- ``max_train`` / ``max_test``: caps for the train / test splits.
- ``max_per_split``: ``{split_name: int}`` for arbitrary splits; merged with the
  shortcuts above (``max_train``/``max_test`` win on conflict).
- ``strategy`` (default ``"head"``): ``"head"`` keeps the first N rows;
  ``"random"`` shuffles deterministically with ``seed`` then takes N.
- ``seed`` (default ``42``): used by the ``"random"`` strategy.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox


_VALID_STRATEGIES = {"head", "random"}


def _find_latest_dataset(inputs: list[Artifact]) -> DatasetArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, DatasetArtifact):
            return artifact
    raise ValueError("sample requires at least one DatasetArtifact input")


def _resolve_caps(spec: dict[str, Any]) -> dict[str, int]:
    caps: dict[str, int] = {}
    raw = spec.get("max_per_split") or {}
    if not isinstance(raw, dict):
        raise ValueError("sample: 'max_per_split' must be a dict[str, int]")
    for split, value in raw.items():
        caps[str(split)] = int(value)
    if "max_train" in spec and spec["max_train"] is not None:
        caps["train"] = int(spec["max_train"])
    if "max_test" in spec and spec["max_test"] is not None:
        caps["test"] = int(spec["max_test"])
    return caps


def _sample_split(dataset: Dataset, cap: int, strategy: str, seed: int) -> tuple[Dataset, int]:
    if cap < 0:
        raise ValueError("sample: cap must be >= 0")
    if dataset.num_rows <= cap:
        return dataset, 0
    if strategy == "head":
        sampled = dataset.select(range(cap))
    elif strategy == "random":
        shuffled = dataset.shuffle(seed=seed)
        sampled = shuffled.select(range(cap))
    else:  # pragma: no cover - validated upstream
        raise ValueError(f"sample: unsupported strategy '{strategy}'")
    return sampled, dataset.num_rows - cap


class SampleOperator(Operator):
    name = "sample"
    description = "Deterministically cap the size of one or more dataset splits."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = _find_latest_dataset(inputs)
        strategy = str(kwargs.get("strategy", "head"))
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(f"sample: strategy must be one of {sorted(_VALID_STRATEGIES)}")
        seed = int(kwargs.get("seed", 42))
        caps = _resolve_caps(kwargs)
        if not caps:
            raise ValueError("sample requires at least one of max_train/max_test/max_per_split")

        dataset_obj = source.load()
        removed_counts: dict[str, int] = {}
        kept_counts: dict[str, int] = {}

        if isinstance(dataset_obj, DatasetDict):
            sampled_splits: dict[str, Dataset] = {}
            for split, split_dataset in dataset_obj.items():
                if split in caps:
                    sampled_splits[split], removed_counts[split] = _sample_split(
                        split_dataset, caps[split], strategy, seed
                    )
                else:
                    sampled_splits[split] = split_dataset
                    removed_counts[split] = 0
                kept_counts[split] = sampled_splits[split].num_rows
            sampled: Dataset | DatasetDict = DatasetDict(sampled_splits)
        else:
            cap = caps.get("train")
            if cap is None:
                raise ValueError(
                    "sample: dataset has no splits; provide max_train (or max_per_split with 'train')"
                )
            sampled_dataset, removed = _sample_split(dataset_obj, cap, strategy, seed)
            sampled = sampled_dataset
            removed_counts = {"train": removed}
            kept_counts = {"train": sampled_dataset.num_rows}

        artifact_id = make_artifact_id("dataset")
        target = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-sample",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={
                "sample_strategy": strategy,
                "sample_seed": seed,
                "sample_caps": caps,
                "sample_removed_counts": removed_counts,
                "sample_kept_counts": kept_counts,
            },
        )
        target.save(sampled)
        sandbox.log(
            "sample",
            f"source={source.artifact_id}\noutput={target.artifact_id}\n"
            f"strategy={strategy}\nseed={seed}\ncaps={caps}\nkept={kept_counts}\n",
        )
        return target
