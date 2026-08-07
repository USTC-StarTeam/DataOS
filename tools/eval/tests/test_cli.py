"""dataos-eval: the sbatch header is code-owned (single node, env pool, gres) — agents
only name the task/model/output. Tests cover the render + a fake-sbatch submit + collect."""

import json
import stat
from pathlib import Path

from dataos_eval.cli import main

REPO = Path(__file__).resolve().parents[3]  # the DataOS repo root (configs/ lives there)


def _submit_args(tmp_path, **over):
    argv = {
        "repo": str(REPO), "environment": "PTB_H100_slurm", "task_config": "posttrainbench",
        "task": "gsm8k", "model_path": "/models/base", "model_name": "baseline",
        "output_dir": str(tmp_path / "eval" / "gsm8k_limit-1"),
        "run_root": str(tmp_path), "limit": -1, "gpus": 1, "cpus": 8, "dry_run": False,
    }
    argv.update(over)
    return argv


def _run_cli(argv: list[str]) -> int:
    import sys

    old = sys.argv
    sys.argv = ["dataos-eval"] + argv
    try:
        main()
    except SystemExit as e:
        return e.code or 0
    finally:
        sys.argv = old
    return 0


def test_submit_dry_run_bakes_everything(tmp_path, capsys):
    rc = _run_cli(["submit", *_argv(_submit_args(tmp_path, dry_run=True))])
    assert rc == 0
    out = capsys.readouterr().out
    assert "-N1" in out and "-n1" in out                       # single node, single task
    assert "--nodelist=gpu01,gpu02,gpu03" in out               # the env node pool
    assert "--gres=gpu:1" in out
    assert "run_lmeval_eval.py" in out and "--limit -1" in out
    assert "TRITON_LIBCUDA_PATH" in out and "HF_HUB_OFFLINE=1" in out


def test_submit_writes_handle_with_fake_sbatch(tmp_path, monkeypatch, capsys):
    fake = tmp_path / "sbatch"
    fake.write_text("#!/bin/bash\necho 424242\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    rc = _run_cli(["submit", *_argv(_submit_args(tmp_path))])
    assert rc == 0
    handle = json.loads((tmp_path / "eval_handle.json").read_text())
    assert handle["job_id"] == "424242"
    assert handle["needs_judge"] is False
    assert json.loads(capsys.readouterr().out)["job_id"] == "424242"


def test_collect_reads_score(tmp_path, capsys):
    out = tmp_path / "eval" / "gsm8k_limit-1"
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps({"score": 66.6}))
    (tmp_path / "eval_handle.json").write_text(json.dumps(
        {"job_id": "1", "task": "gsm8k", "output_dir": str(out), "needs_judge": False}))
    rc = _run_cli(["collect", "--repo", str(REPO), str(tmp_path / "eval_handle.json")])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["score"] == 66.6


def test_collect_fails_clean_without_score(tmp_path, capsys):
    (tmp_path / "eval_handle.json").write_text(json.dumps(
        {"job_id": "1", "task": "gsm8k", "output_dir": str(tmp_path / "nope"),
         "needs_judge": False}))
    rc = _run_cli(["collect", "--repo", str(REPO), str(tmp_path / "eval_handle.json")])
    assert rc == 1


def _argv(d: dict) -> list[str]:
    argv = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k.replace('_', '-')}")
        else:
            argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv
