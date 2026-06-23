"""``dataos.load`` — the loading half of the data stage.

A thin router on top of :class:`dataos_operator.Dataset` that fills the one gap
the operator library does not cover: Hugging Face ``save_to_disk`` directories
(Arrow ``Dataset`` / ``DatasetDict`` dumps, such as the bundled
``resources/data/datasets/gsm8k``). Everything else delegates straight to the operator
loaders so a planner has a single entry point::

    dataos.load("hf://openai/gsm8k", split="train", limit=100)   # HF hub
    dataos.load("data/train.jsonl")                              # local file
    dataos.load("resources/data/datasets/gsm8k", split="train", limit=8)        # save_to_disk dir
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dataos_operator import Dataset

_LOCAL_FILE_SUFFIXES = {".jsonl", ".ndjson", ".json", ".csv"}


def load(
    source: str | Path,
    *,
    split: str = "train",
    field_map: dict[str, str] | None = None,
    limit: int | None = None,
    revision: str | None = None,
) -> "Dataset":
    """Load a dataset into a :class:`dataos_operator.Dataset`.

    Routing:
      * ``hf://<repo>`` (or ``hf://<repo>/<name>``) -> ``Dataset.from_hf``.
      * a local ``.jsonl`` / ``.json`` / ``.csv`` file -> ``Dataset.from_local``.
      * a ``save_to_disk`` directory (``datasets.load_from_disk``-readable) ->
        select ``split`` and build via ``Dataset.from_records``.

    ``revision`` pins a Hugging Face dataset commit/tag/branch (forwarded to
    ``load_dataset``) for reproducibility; it only applies to ``hf://`` sources.
    """
    from dataos_operator import Dataset

    source_text = str(source)
    if source_text.startswith("hf://"):
        return Dataset.from_hf(
            source_text.removeprefix("hf://"),
            split=split,
            field_map=field_map,
            limit=limit,
            revision=revision,
        )

    source_path = Path(source).expanduser()
    if source_path.is_file() and source_path.suffix.lower() in _LOCAL_FILE_SUFFIXES:
        return Dataset.from_local(source_path, field_map=field_map, limit=limit)

    if source_path.is_dir():
        return _load_save_to_disk(source_path, split=split, field_map=field_map, limit=limit)

    raise ValueError(
        f"Cannot load dataset source {source_text!r}: expected an 'hf://' path, a local "
        ".jsonl/.json/.csv file, or a Hugging Face save_to_disk directory."
    )


def _load_save_to_disk(
    source_path: Path,
    *,
    split: str,
    field_map: dict[str, str] | None,
    limit: int | None,
) -> "Dataset":
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict, load_from_disk

    from dataos_operator import Dataset

    try:
        loaded = load_from_disk(str(source_path))
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise ValueError(
            f"Not a readable Hugging Face save_to_disk directory: {source_path}"
        ) from exc

    if isinstance(loaded, DatasetDict):
        if split not in loaded:
            available = ", ".join(sorted(loaded.keys()))
            raise ValueError(
                f"Split {split!r} not found in {source_path}; available splits: {available}"
            )
        hf_dataset = loaded[split]
    elif isinstance(loaded, HFDataset):
        hf_dataset = loaded
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported Hugging Face object at {source_path}: {type(loaded)!r}")

    if limit is not None:
        hf_dataset = hf_dataset.select(range(min(limit, hf_dataset.num_rows)))

    records: list[dict[str, Any]] = hf_dataset.to_list()
    if field_map is not None:
        records = [
            {target: row[source_field] for target, source_field in field_map.items()}
            for row in records
        ]

    # dataos_operator's save_to_disk writes a sidecar with e.g. dataset_format — carry it
    # through so a prepared dataset injected via run_one --data keeps its format provenance.
    metadata: dict[str, Any] = {}
    sidecar = source_path / "dataos_metadata.json"
    if sidecar.is_file():
        try:
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                metadata.update(saved)
        except (OSError, json.JSONDecodeError):
            pass
    metadata.update(
        {
            "source": source_path.name,
            "loader": "save_to_disk",
            "path": str(source_path),
            "split": split,
            "records": len(records),
        }
    )
    return Dataset.from_records(records, metadata=metadata)
