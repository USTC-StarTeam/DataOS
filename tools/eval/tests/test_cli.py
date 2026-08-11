"""dataos-eval: the sbatch header is code-owned (single node, env pool, gres) — agents
only name the task/model/output. Tests cover the render + a fake-sbatch submit + collect."""

import json
import stat
from pathlib import Path

from dataos_eval.cli import main

REPO = Path(__file__).resolve().parents[3]  # the DataOS repo root (configs/ lives there)


def _model_dir(tmp_path, model_type: str = "qwen3") -> str:
    """A minimal HF model dir. The chat template is resolved from THIS file's model_type — not
    from --model-name/--model-path substrings, which the caller controls."""
    import json

    d = tmp_path / f"model-{model_type}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return str(d)


def _submit_args(tmp_path, **over):
    argv = {
        "repo": str(REPO), "environment": "PTB_H100_slurm", "task_config": "posttrainbench",
        "task": "gsm8k", "model_path": _model_dir(tmp_path), "model_name": "baseline",
        "output_dir": str(tmp_path / "eval" / "gsm8k_limit-1"),
        "run_root": str(tmp_path), "limit": -1, "gpus": 1, "cpus": 8, "time": "12:00:00",
        "dry_run": False,
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
    assert "--time=12:00:00" in out                             # explicit: no --time pends forever (QOS 7d > partition MaxTime 3d)
    assert "run_lmeval_eval.py" in out and "--limit -1" in out
    assert "TRITON_LIBCUDA_PATH" in out and "HF_HUB_OFFLINE=1" in out


def test_submit_clamps_time_to_the_partition_max(tmp_path, capsys):
    rc = _run_cli(["submit", *_argv(_submit_args(tmp_path, time="7-00:00:00", dry_run=True))])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--time=3-00:00:00" in out      # clamped to env slurm.max_time (a 7d ask pends forever)
    assert "--time=7-00:00:00" not in out


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


def test_map_state_tolerates_suffixes_and_fail_states():
    from dataos_eval.cli import _map_state

    assert _map_state("COMPLETED") == "COMPLETED"
    assert _map_state("CANCELLED by 1003") == "FAILED"      # scancel (e.g. GPU Guard)
    assert _map_state("NODE_FAIL") == "FAILED"
    assert _map_state("OUT_OF_MEMORY") == "FAILED"
    assert _map_state("PREEMPTED") == "FAILED"
    assert _map_state("TIMEOUT") == "TIMEOUT"
    assert _map_state("RUNNING") == "RUNNING"
    assert _map_state("") == "PENDING"


def test_status_reads_the_primary_job_row(tmp_path, monkeypatch, capsys):
    """sacct lists the main job first, then .batch/.extern — a failed job's .extern is
    COMPLETED, so reading the LAST row misreports FAILED as COMPLETED."""
    import stat as _stat

    from dataos_eval.cli import _state

    fake = tmp_path / "sacct"
    fake.write_text("#!/bin/bash\necho FAILED; echo COMPLETED; echo COMPLETED\n")
    fake.chmod(fake.stat().st_mode | _stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    assert _state("1") == "FAILED"


def test_render_quotes_paren_paths(tmp_path, capsys):
    """Fresh-instance run roots can contain parens ('ptb-x-1' style dirs) — the generated
    job script must quote every interpolated path or bash dies at line 6."""
    rc = _run_cli(["submit", *_argv(_submit_args(tmp_path, run_root=str(tmp_path / "ptb-x(1)"),
                                                output_dir=str(tmp_path / "ptb-x(1)/eval"),
                                                dry_run=True))])
    assert rc == 0
    out = capsys.readouterr().out
    script = out[out.find("#!/bin/bash"):]
    import subprocess as sp
    r = sp.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert r.returncode == 0, r.stderr  # the script parses


# --- protocol resolution: config + the model itself, never the caller's label ----------------


def test_chat_template_comes_from_the_models_model_type(tmp_path, capsys):
    """smol3 showed why: --model-name is filled by the experimenter, so guessing the template
    from it let an agent change the eval protocol by choosing a label."""
    args = _submit_args(tmp_path, model_path=_model_dir(tmp_path, "smollm3"),
                        model_name="totally-misleading-qwen-name", dry_run=True)
    assert _run_cli(["submit", *_argv(args)]) == 0
    out = capsys.readouterr().out
    assert "--chat-template" in out
    assert "smollm.jinja" in out            # from model_type, not from the qwen-ish label
    assert "qwen3.jinja" not in out


def test_unknown_model_family_is_refused(tmp_path):
    rc = _run_cli(["submit", *_argv(_submit_args(
        tmp_path, model_path=_model_dir(tmp_path, "llama"), dry_run=True))])
    assert isinstance(rc, str) and "no chat template declared" in rc


def test_judge_flag_comes_from_the_protocol_not_a_tool_table(tmp_path):
    """`JUDGE_BENCHES` used to be hardcoded in cli.py — bench knowledge inside the tool."""
    import yaml

    from dataos_eval.cli import _load

    _, _, protocol = _load(REPO, "PTB_H100_slurm", "posttrainbench")
    declared = {b for b, c in protocol["benches"].items() if c["judge"]}
    assert declared == {"healthbench", "arenahardwriting"}
    assert "JUDGE_BENCHES" not in (
        (REPO / "tools" / "eval" / "src" / "dataos_eval" / "cli.py").read_text(encoding="utf-8")
    )
    del yaml
