"""A run that cannot fit its own output must be refused before it burns the allocation.

The 4b2 v5 batch exhausted /data2 mid-run: training completed, then ``save_pretrained`` failed
writing the weights. Hours of H100 time produced nothing, and the error read as a size bug
("File too large (os error 27)") rather than a full filesystem, so it cost debugging time twice.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dataos_train import precheck
from dataos_train.spec import SFT, DatasetRef, EnvSpec, TrainSpec

GB = 1024**3


def _model_dir(tmp_path: Path, gib: float) -> str:
    d = tmp_path / "base"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"model_type": "qwen3"}', encoding="utf-8")
    (d / "model.safetensors").write_bytes(b"\0" * 1024)  # size is faked below, not written
    return str(d)


def _spec(tmp_path: Path, base: str) -> TrainSpec:
    return TrainSpec(
        name="e0",
        base_model=base,
        run_dir=str(tmp_path / "runs" / "e0"),  # deliberately not created yet
        dataset=DatasetRef(path=str(tmp_path / "data")),
        method=SFT(template="qwen3"),
        env=EnvSpec(backends={"llamafactory": "/envs/Llama"}),
    )


@pytest.fixture
def fake_space(monkeypatch):
    """Pin free space and the model's weight size; the real numbers are machine-dependent."""

    def _apply(*, free_gib: float, model_gib: float):
        monkeypatch.setattr(
            precheck.shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(0, 0, int(free_gib * GB))
        )
        monkeypatch.setattr(precheck, "_model_bytes", lambda _p: int(model_gib * GB))

    return _apply


def test_a_run_that_cannot_fit_its_output_is_refused(tmp_path, fake_space) -> None:
    fake_space(free_gib=5, model_gib=8)  # needs 12 GiB at 1.5x, has 5
    errors = precheck.precheck_disk(_spec(tmp_path, _model_dir(tmp_path, 8)))
    assert len(errors) == 1
    assert "5.0 GiB free" in errors[0] and "12.0 GiB" in errors[0]


def test_enough_space_passes(tmp_path, fake_space) -> None:
    fake_space(free_gib=40, model_gib=8)
    assert precheck.precheck_disk(_spec(tmp_path, _model_dir(tmp_path, 8))) == []


def test_an_unreadable_base_model_is_not_an_error(tmp_path) -> None:
    """The check compares against the base model's size; with no readable model there is nothing
    to compare, and refusing would break every spec built where the model path is not visible."""
    assert precheck.precheck_disk(_spec(tmp_path, "/not/on/this/host")) == []


def test_it_probes_an_existing_ancestor_of_run_dir(tmp_path, monkeypatch) -> None:
    """run_dir is created by the job, so the check must walk up to a directory that exists —
    otherwise it would crash on the very first run under a fresh run root."""
    seen: list[Path] = []
    monkeypatch.setattr(precheck, "_model_bytes", lambda _p: 8 * GB)
    monkeypatch.setattr(
        precheck.shutil,
        "disk_usage",
        lambda p: seen.append(Path(p)) or shutil._ntuple_diskusage(0, 0, 40 * GB),
    )
    assert precheck.precheck_disk(_spec(tmp_path, _model_dir(tmp_path, 8))) == []
    assert seen and seen[0].exists(), f"probed a path that does not exist: {seen}"


def test_retained_checkpoints_raise_the_requirement(tmp_path, fake_space) -> None:
    """save_steps keeps whole models on disk; the precheck has to count them or it green-lights
    exactly the run most likely to fill the filesystem."""
    fake_space(free_gib=20, model_gib=8)  # 1.5x = 12 GiB fits; 1.5+3 = 36 GiB does not
    spec = _spec(tmp_path, _model_dir(tmp_path, 8))
    assert precheck.precheck_disk(spec) == []

    spec.method.finetuning_type = "full"
    spec.method.save_steps = 100
    spec.method.save_total_limit = 3
    errors = precheck.precheck_disk(spec)
    assert len(errors) == 1
    assert "36.0 GiB" in errors[0] and "3 retained checkpoints" in errors[0]


def test_lora_checkpoints_do_not_raise_the_requirement(tmp_path, fake_space) -> None:
    """A LoRA checkpoint is an adapter, not a model — counting it as one would refuse runs that
    fit comfortably."""
    fake_space(free_gib=20, model_gib=8)
    spec = _spec(tmp_path, _model_dir(tmp_path, 8))
    spec.method.finetuning_type = "lora"
    spec.method.save_steps = 100
    spec.method.save_total_limit = 5
    assert precheck.precheck_disk(spec) == []
