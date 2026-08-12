"""Cutoff-vs-data-length heuristic warning in LF backend validate()."""

from __future__ import annotations

from conftest import MODEL_DIR

import json
import warnings

import pytest

from dataos_train.backends.llamafactory import LlamaFactoryBackend
from dataos_train.spec import SFT, DatasetRef, TrainSpec


def _spec_with_data(path, cutoff) -> TrainSpec:  # noqa: ANN001
    return TrainSpec(
        name="exp",
        base_model=MODEL_DIR,
        run_dir="/r",
        dataset=DatasetRef(path=str(path)),
        method=SFT(finetuning_type="lora", template="qwen3", cutoff_len=cutoff),
    )


def _write_jsonl(path, lengths) -> None:  # noqa: ANN001
    with open(path, "w") as f:
        for n in lengths:
            f.write(json.dumps({"instruction": "q", "input": "", "output": "x" * n}) + "\n")


def test_cutoff_warning_fires_when_many_samples_too_long(tmp_path) -> None:
    data = tmp_path / "data.jsonl"
    _write_jsonl(data, [100, 100, 4000, 4000, 4000])  # 60% over cutoff*3 chars
    with pytest.warns(UserWarning, match="truncation teaches non-termination"):
        LlamaFactoryBackend().validate(_spec_with_data(data, 1024))


def test_cutoff_warning_quiet_when_data_fits(tmp_path) -> None:
    data = tmp_path / "data.jsonl"
    _write_jsonl(data, [100, 200, 300, 400])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LlamaFactoryBackend().validate(_spec_with_data(data, 1024))
    assert not any("truncation" in str(w.message) for w in caught)


def test_cutoff_warning_messages_format(tmp_path) -> None:
    data = tmp_path / "data.jsonl"
    with open(data, "w") as f:
        for _ in range(10):
            rec = {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "x" * 5000},
                ]
            }
            f.write(json.dumps(rec) + "\n")
    with pytest.warns(UserWarning, match="truncation"):
        LlamaFactoryBackend().validate(_spec_with_data(data, 1024))
