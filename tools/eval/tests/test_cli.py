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
        "nodelist": None, "qos": "gpu_priv", "dry_run": False,
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


def _dry_run_spec(tmp_path, capsys, **over) -> dict:
    """The JobSpec a submit would hand the launcher. Since the launcher registry landed, this —
    not a rendered sbatch — is what dataos-eval owns and what the caller cannot reach past."""
    assert _run_cli(["submit", *_argv(_submit_args(tmp_path, dry_run=True, **over))]) == 0
    return json.loads(capsys.readouterr().out)


def test_submit_dry_run_bakes_everything(tmp_path, capsys):
    spec = _dry_run_spec(tmp_path, capsys)
    assert spec["launcher"] == "slurm"
    job = spec["job"]
    res = job["resources"]
    assert (res["nodes"], res["gpus"], res["cpus"]) == (1, 1, 8)   # single node, one GPU
    assert job["slurm"]["nodelist"] == "gpu01,gpu02,gpu03"      # the env node pool
    assert "--time=12:00:00" in job["slurm"]["extra_sbatch"]    # explicit: no --time pends forever
    cmd = " ".join(job["command"])
    assert "run_lmeval_eval.py" in cmd and "--limit -1" in cmd
    assert job["env"]["HF_HUB_OFFLINE"] == "1"
    assert any("TRITON_LIBCUDA_PATH" in c for c in job["pre_cmds"])


def test_submit_clamps_time_to_the_partition_max(tmp_path, capsys):
    extra = _dry_run_spec(tmp_path, capsys, time="7-00:00:00")["job"]["slurm"]["extra_sbatch"]
    # clamped to env slurm.max_time (a 7d ask pends forever)
    assert "--time=3-00:00:00" in extra
    assert "--time=7-00:00:00" not in extra


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
        if v is None:          # an optional flag the caller omits, not the string "None"
            continue
        if isinstance(v, bool):
            if v:
                argv.append(f"--{k.replace('_', '-')}")
        else:
            argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv


# The sacct state mapping and the first-row rule this file used to test moved into
# dataos-launch with the launcher; they are covered by tools/train/tests/test_slurm_render.py.


def test_render_quotes_paren_paths(tmp_path, capsys):
    """Fresh-instance run roots can contain parens ('ptb-x-1' style dirs) — the generated
    job script must quote every interpolated path or bash dies at line 6. Rendered here through
    the launcher the CLI would hand this spec to; asserting on the dry-run text instead let this
    test pass vacuously once dry-run started printing the spec rather than the script."""
    import subprocess as sp

    from dataos_launch.launchers.slurm import SlurmLauncher
    from dataos_launch.types import JobSpec

    spec = _dry_run_spec(tmp_path, capsys, run_root=str(tmp_path / "ptb-x(1)"),
                         output_dir=str(tmp_path / "ptb-x(1)/eval"))
    script = SlurmLauncher().render_sbatch(JobSpec.model_validate(spec["job"]))
    assert script.startswith("#!/bin/bash")
    assert "ptb-x(1)" in script                                  # the risky path really is in there
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


def test_a_run_node_pin_overrides_the_env_node_pool(tmp_path, capsys):
    """The run's pin wins — the precedence dataos.agent.train_spec already uses.

    `env.slurm.nodes` is not a pin, it is a POOL: `--nodelist=gpu01,gpu02,gpu03` legally means
    "any of these", and Slurm obliged. A run pinned to gpu02 put 35 of 35 training jobs on gpu02
    and 68 of 94 eval jobs elsewhere, because this was the one submitter that never saw the pin.
    """
    job = _dry_run_spec(tmp_path, capsys, nodelist="gpu02")["job"]
    assert job["slurm"]["nodelist"] == "gpu02"


def test_without_a_pin_it_still_falls_back_to_the_env_pool(tmp_path, capsys):
    job = _dry_run_spec(tmp_path, capsys)["job"]
    assert job["slurm"]["nodelist"] == "gpu01,gpu02,gpu03"


def test_the_run_chooses_the_qos(tmp_path, capsys):
    """`qos` was env-only until it moved to the run config, at which point the field this read
    simply vanished: jobs went out with no --qos and Slurm applied the account default — the
    GUARDED queue — while the job's own prompt said it was unguarded. Nothing errored."""
    job = _dry_run_spec(tmp_path, capsys, qos="gpu_normal")["job"]
    assert job["slurm"]["qos"] == "gpu_normal"


def test_the_runs_gpus_and_cpus_are_not_defaults(tmp_path, capsys):
    """Latent half of the same gap: the caller passed neither, so both fell to argparse
    defaults — and every run config happened to ask for exactly those (1 and 8). The first run
    to ask for 2 GPUs would have trained on 2 and evaluated on 1."""
    job = _dry_run_spec(tmp_path, capsys, gpus=2, cpus=11)["job"]
    assert job["resources"]["gpus"] == 2
    assert job["slurm"]["cpus_per_task"] == 11
