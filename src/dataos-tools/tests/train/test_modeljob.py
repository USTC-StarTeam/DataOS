"""ModelJob: spec validation, build rendering, collect parsing — all torch-free."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from dataos_launch.types import Status
from pydantic import ValidationError

import dataos_train.cli as cli
import dataos_train.modeljob as mj
from dataos_train.modeljob import score_entry


def _spec_dict(**kw) -> dict:
    base = {
        "name": "l2",
        "model": "/models/qwen",
        "dataset": {"path": "/data/train.jsonl", "format": "alpaca"},
        "interaction": {"kind": "loss"},
    }
    return base | kw


def _spec(**kw) -> mj.ModelJobSpec:
    return mj.ModelJobSpec.model_validate(_spec_dict(**kw))


# --- spec -----------------------------------------------------------------


def test_spec_requires_model() -> None:
    data = _spec_dict()
    del data["model"]
    with pytest.raises(ValidationError):
        mj.ModelJobSpec.model_validate(data)
    assert any("model is required" in e for e in mj.validate(_spec(model="")))


def test_spec_requires_dataset() -> None:
    data = _spec_dict()
    del data["dataset"]
    with pytest.raises(ValidationError):
        mj.ModelJobSpec.model_validate(data)
    errors = mj.validate(_spec(dataset={"path": "", "format": "alpaca"}))
    assert any("dataset.path is required" in e for e in errors)


def test_spec_rejects_bad_kind_and_extra_keys() -> None:
    with pytest.raises(ValidationError):
        _spec(interaction={"kind": "perplexity"})
    with pytest.raises(ValidationError):
        mj.ModelJobSpec.model_validate(_spec_dict(bogus=1))


def test_spec_rejects_bad_batch_and_max_len() -> None:
    with pytest.raises(ValidationError):
        _spec(interaction={"kind": "loss", "batch_size": 0})
    with pytest.raises(ValidationError):
        _spec(interaction={"kind": "loss", "max_len": -1})
    # belt-and-braces: a spec bypassing schema validation is still rejected by validate()
    bypassed = mj.LossInteraction.model_construct(kind="loss", batch_size=0, max_len=0)
    errors = mj.validate(_spec(interaction=bypassed))
    assert any("batch_size" in e for e in errors)
    assert any("max_len" in e for e in errors)


def test_validate_rejects_unscorable_format_and_bad_kind_bypass() -> None:
    errors = mj.validate(_spec(dataset={"path": "/d", "format": "rl"}))
    assert any("not scorable" in e for e in errors)
    # schema bypass: the discriminated union rejects a bad kind at parse time, so sneak
    # it past the schema and check validate() still guards it
    spec = _spec()
    spec.interaction = mj.LossInteraction.model_construct(kind="perplexity")
    errors = mj.validate(spec)
    assert any("unsupported interaction kind" in e for e in errors)
    assert mj.validate(_spec()) == []


# --- build ------------------------------------------------------------------


def test_build_renders_run_score_sh() -> None:
    spec = _spec()
    job = mj.build(spec, run_dir="/runs/l2")
    script = job.assets["run_score.sh"]
    assert "cd /runs/l2" in script
    assert "python -m dataos_train.modeljob.score_entry --config modeljob.json" in script
    assert job.command == ["bash", "run_score.sh"]
    assert job.backend == "modeljob"
    cfg = json.loads(job.assets["modeljob.json"])
    assert cfg["dataset"]["path"] == "/data/train.jsonl"
    assert cfg["interaction"]["output_key"] == "metadata.runtime.loss"
    assert cfg["model"] == "/models/qwen"


def test_build_run_dir_default_and_env_resolution() -> None:
    job = mj.build(_spec())
    assert job.run_dir == "/data/modeljobs/l2"  # file dataset -> sibling modeljobs/<name>
    job_dir = mj.build(_spec(dataset={"path": "/data/curated", "format": "messages"}))
    assert job_dir.run_dir == "/data/curated/modeljobs/l2"
    spec = _spec(
        env={"conda_sh": "/c/conda.sh", "backends": {"modeljob": "train-env"}},
    )
    job = mj.build(spec, run_dir="/r")
    assert job.conda_sh == "/c/conda.sh"
    assert job.conda_env == "train-env"
    assert job.resources.gpus == 1   # placement (partition/gres) is the launcher's, not the Job's


# --- collect ----------------------------------------------------------------


def _write_run(run_dir: Path, scores: list[dict], n_records: int = 3) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = run_dir / "data.jsonl"
    records = [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(n_records)]
    data_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    (run_dir / "modeljob.json").write_text(
        json.dumps(_spec_dict(dataset={"path": str(data_path), "format": "alpaca"})),
        encoding="utf-8",
    )
    (run_dir / "scores.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in scores), encoding="utf-8"
    )


def _score_lines(losses: list) -> list[dict]:
    return [{"index": i, "key": "metadata.runtime.loss", "loss": v} for i, v in enumerate(losses)]


def test_collect_parses_scores_and_stats(tmp_path: Path) -> None:
    _write_run(tmp_path, _score_lines([1.0, 2.0, 3.0]))
    result = mj.collect(str(tmp_path))
    assert result.status == Status.COMPLETED
    assert result.rows == 3 and result.key == "metadata.runtime.loss"
    m = result.metrics
    assert m["n_scored"] == 3 and m["n_null"] == 0
    assert m["loss_mean"] == pytest.approx(2.0)
    assert m["loss_min"] == 1.0 and m["loss_max"] == 3.0 and m["loss_p50"] == 2.0


def test_collect_missing_file_fails(tmp_path: Path) -> None:
    result = mj.collect(str(tmp_path))
    assert result.status == Status.FAILED
    assert "scores.jsonl not found" in (result.error or "")


def test_collect_rejects_non_finite_loss(tmp_path: Path) -> None:
    _write_run(tmp_path, _score_lines([1.0, float("nan"), 3.0]))
    result = mj.collect(str(tmp_path))
    assert result.status == Status.FAILED
    assert "non-finite loss" in (result.error or "")


def test_collect_rejects_row_count_mismatch(tmp_path: Path) -> None:
    _write_run(tmp_path, _score_lines([1.0, 2.0]), n_records=3)
    result = mj.collect(str(tmp_path))
    assert result.status == Status.FAILED
    assert "3 records" in (result.error or "")


def test_collect_rejects_gapped_indices_and_mixed_keys(tmp_path: Path) -> None:
    _write_run(tmp_path, _score_lines([1.0, 2.0, 3.0]))
    lines = _score_lines([1.0, 2.0, 3.0])
    lines[1]["index"] = 7
    (tmp_path / "scores.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in lines), encoding="utf-8"
    )
    assert mj.collect(str(tmp_path)).status == Status.FAILED
    lines = _score_lines([1.0, 2.0, 3.0])
    lines[2]["key"] = "metadata.other"
    (tmp_path / "scores.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in lines), encoding="utf-8"
    )
    assert mj.collect(str(tmp_path)).status == Status.FAILED


def test_collect_tolerates_null_loss_but_counts_it(tmp_path: Path) -> None:
    _write_run(tmp_path, _score_lines([1.0, None, 3.0]))
    result = mj.collect(str(tmp_path))
    assert result.status == Status.COMPLETED
    assert result.metrics["n_null"] == 1 and result.metrics["n_scored"] == 2
    assert result.metrics["loss_mean"] == pytest.approx(2.0)


# --- CLI ----------------------------------------------------------------------


def test_cli_validate_build_collect(tmp_path: Path, capsys) -> None:  # noqa: ANN001
    spec_path = tmp_path / "modeljob_spec.json"
    run_dir = tmp_path / "run"
    data_path = tmp_path / "data.jsonl"
    data_path.write_text('{"instruction": "q", "output": "a"}\n', encoding="utf-8")
    spec_path.write_text(
        json.dumps(_spec_dict(dataset={"path": str(data_path), "format": "alpaca"})),
        encoding="utf-8",
    )
    assert cli.main(["modeljob-validate", str(spec_path)]) == 0
    assert cli.main(["modeljob-build", str(spec_path), "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "job.json").is_file()  # the rendered Job, as train build writes it
    assert (run_dir / "modeljob.json").is_file()
    assert (run_dir / "run_score.sh").is_file()
    (run_dir / "scores.jsonl").write_text(
        '{"index": 0, "key": "metadata.runtime.loss", "loss": 0.5}\n', encoding="utf-8"
    )
    assert cli.main(["modeljob-collect", str(run_dir)]) == 0
    persisted = json.loads((run_dir / "score_result.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "COMPLETED"
    assert persisted["metrics"]["loss_mean"] == pytest.approx(0.5)
    capsys.readouterr()  # keep the test output clean


# --- torch discipline + assembly helpers -------------------------------------


def test_package_import_is_torch_free() -> None:
    import dataos_train.modeljob.score_entry  # noqa: F401

    assert "torch" not in sys.modules  # heavy deps load only inside the scoring job


def test_to_messages_alpaca() -> None:
    msgs = score_entry.to_messages(
        {"instruction": "inst", "input": "in", "output": "out"}, "alpaca"
    )
    assert msgs == [
        {"role": "user", "content": "inst\nin"},
        {"role": "assistant", "content": "out"},
    ]
    msgs = score_entry.to_messages({"instruction": "inst", "output": "out"}, "alpaca")
    assert msgs[0]["content"] == "inst"


def test_to_messages_passthrough_and_sharegpt() -> None:
    rec = {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}
    assert score_entry.to_messages(rec, "messages") == rec["messages"]
    sgpt = {
        "conversations": [
            {"from": "system", "value": "s"},
            {"from": "human", "value": "u"},
            {"from": "gpt", "value": "a"},
        ]
    }
    assert score_entry.to_messages(sgpt, "sharegpt") == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    # auto: shape sniffing per record
    assert score_entry.detect_format(rec) == "messages"
    assert score_entry.detect_format(sgpt) == "sharegpt"
    assert score_entry.detect_format({"instruction": "i", "output": "o"}) == "alpaca"


def test_split_prompt_requires_final_assistant() -> None:
    msgs = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    prompt, completion = score_entry.split_prompt(msgs)
    assert prompt == msgs[:-1] and completion == "a"
    with pytest.raises(ValueError, match="assistant"):
        score_entry.split_prompt([{"role": "user", "content": "u"}])


def test_common_prefix_len() -> None:
    assert score_entry._common_prefix_len([1, 2, 3], [1, 2, 3, 4]) == 3
    assert score_entry._common_prefix_len([1, 9], [1, 2]) == 1
    assert score_entry._common_prefix_len([], [1]) == 0


def test_preamble_activates_venv_by_path_without_conda():
    """uv/standard venvs (absolute path, no conda_sh) must be activated by path."""
    from types import SimpleNamespace

    from dataos_launch.launchers.base import build_preamble

    job = SimpleNamespace(
        conda_sh=None, conda_env="/envs/sft/.venv", pythonpath=[], env={}, pre_cmds=[],
        run_dir="/runs/x", launch=SimpleNamespace(kind="direct", nproc_per_node=None,
        nnodes=None), command=["bash", "run_train.sh"],
        resources=SimpleNamespace(gpus=1, nodes=1),
    )
    lines = build_preamble(job)
    assert any("source /envs/sft/.venv/bin/activate" in line for line in lines), lines

    job2 = SimpleNamespace(**{**vars(job), "conda_env": None})
    assert not any("activate" in line for line in build_preamble(job2))
