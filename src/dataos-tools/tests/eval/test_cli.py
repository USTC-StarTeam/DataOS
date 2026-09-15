"""dataos-eval: the sbatch header is code-owned (single node, env pool, gres) — agents
only name the task/model/output. Tests cover the render + a fake-sbatch submit + collect."""

import json
import stat
from pathlib import Path

from dataos_eval.cli import main

REPO = Path(__file__).resolve().parents[4]  # the DataOS repo root (configs/ lives there)


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
        "repo": str(REPO), "environment": "example-slurm", "task_config": "posttrainbench",
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
    """The Job a submit would hand the launcher. Since the launcher registry landed, this —
    not a rendered sbatch — is what dataos-eval owns and what the caller cannot reach past."""
    assert _run_cli(["submit", *_argv(_submit_args(tmp_path, dry_run=True, **over))]) == 0
    return json.loads(capsys.readouterr().out)


def test_submit_dry_run_bakes_everything(tmp_path, capsys):
    spec = _dry_run_spec(tmp_path, capsys)
    assert spec["launcher"] == "slurm"
    job = spec["job"]
    res = job["resources"]
    assert (res["nodes"], res["gpus"], res["cpus"]) == (1, 1, 8)   # single node, one GPU
    assert spec["placement"]["nodelist"] == "<node-a>,<node-b>,<node-c>"   # the env node pool
    assert res["time_h"] == 12.0   # the launcher always emits --time; no --time pends forever
    cmd = " ".join(job["command"])
    assert "run_lmeval_eval.py" in cmd and "--limit -1" in cmd
    assert job["env"]["HF_HUB_OFFLINE"] == "1"
    assert any("TRITON_LIBCUDA_PATH" in c for c in job["pre_cmds"])


def test_submit_clamps_time_to_the_partition_max(tmp_path, capsys):
    """The ask is a Resources field; the clamp is the launcher's, against the partition MaxTime
    it was configured with. dataos-eval used to clamp too, in its own copy of the parser."""
    from dataos_launch import get_launcher
    from dataos_launch.types import Resources

    spec = _dry_run_spec(tmp_path, capsys, time="7-00:00:00")
    assert spec["job"]["resources"]["time_h"] == 168.0            # asked for 7d
    launcher = get_launcher(spec["launcher"], **spec["placement"])
    assert launcher.time_limit(Resources(**spec["job"]["resources"])) == "72:00:00"


def test_eval_gpu_allocation_reaches_tensor_parallelism(tmp_path, capsys):
    spec = _dry_run_spec(tmp_path, capsys, gpus=8, cpus=64)
    command = spec["job"]["command"]
    assert command[command.index("--tensor-parallel-size") + 1] == "8"
    assert spec["job"]["resources"]["gpus"] == 8


def test_submit_writes_handle_with_fake_sbatch(tmp_path, monkeypatch, capsys):
    fake = tmp_path / "sbatch"
    fake.write_text("#!/bin/bash\necho 424242\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{tmp_path}:/usr/bin:/bin")
    rc = _run_cli(["submit", *_argv(_submit_args(tmp_path))])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["job_id"] == "424242"
    # the record is named after the spec, so a checkpoint sweep leaves one per evaluation
    assert Path(out["handle"]).name == "eval_gsm8k_limit-1.eval_handle.json"
    handle = json.loads(Path(out["handle"]).read_text())
    assert handle["job_id"] == "424242"
    assert handle["needs_judge"] is False


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
# dataos-launch with the launcher; they are covered by tools/tests/launch/test_slurm.py.


def test_render_quotes_paren_paths(tmp_path, capsys):
    """Fresh-instance run roots can contain parens ('ptb-x-1' style dirs) — the generated
    job script must quote every interpolated path or bash dies at line 6. Rendered here through
    the launcher the CLI would hand this spec to; asserting on the dry-run text instead let this
    test pass vacuously once dry-run started printing the spec rather than the script."""
    import subprocess as sp

    from dataos_launch import get_launcher
    from dataos_launch.types import Job

    spec = _dry_run_spec(tmp_path, capsys, run_root=str(tmp_path / "ptb-x(1)"),
                         output_dir=str(tmp_path / "ptb-x(1)/eval"))
    launcher = get_launcher(spec["launcher"], **spec["placement"])
    script = launcher.render(Job.model_validate(spec["job"]))
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
    from dataos_eval.stage import EvalStage

    stage = EvalStage.from_repo(REPO, "example-slurm", "posttrainbench")
    declared = {b for b, c in stage.benches.items() if c["judge"]}
    assert declared == {"healthbench", "arenahardwriting"}
    src = REPO / "src" / "dataos-tools" / "src" / "dataos_eval"
    assert "JUDGE_BENCHES" not in (src / "cli.py").read_text(encoding="utf-8")
    assert "JUDGE_BENCHES" not in (src / "stage.py").read_text(encoding="utf-8")


def test_a_run_node_pin_overrides_the_env_node_pool(tmp_path, capsys):
    """The run's pin wins — the precedence dataos.agent.train_spec already uses.

    `env.slurm.nodes` is not a pin, it is a POOL: `--nodelist=gpu01,gpu02,gpu03` legally means
    "any of these", and Slurm obliged. A run pinned to gpu02 put 35 of 35 training jobs on gpu02
    and 68 of 94 eval jobs elsewhere, because this was the one submitter that never saw the pin.
    """
    assert _dry_run_spec(tmp_path, capsys, nodelist="gpu02")["placement"]["nodelist"] == "gpu02"


def test_without_a_pin_it_still_falls_back_to_the_env_pool(tmp_path, capsys):
    spec = _dry_run_spec(tmp_path, capsys)
    assert spec["placement"]["nodelist"] == "<node-a>,<node-b>,<node-c>"


def test_the_run_chooses_the_qos(tmp_path, capsys):
    """`qos` was env-only until it moved to the run config, at which point the field this read
    simply vanished: jobs went out with no --qos and Slurm applied the account default — the
    GUARDED queue — while the job's own prompt said it was unguarded. Nothing errored."""
    assert _dry_run_spec(tmp_path, capsys, qos="gpu_normal")["placement"]["qos"] == "gpu_normal"


def test_the_runs_gpus_and_cpus_are_not_defaults(tmp_path, capsys):
    """Latent half of the same gap: the caller passed neither, so both fell to argparse
    defaults — and every run config happened to ask for exactly those (1 and 8). The first run
    to ask for 2 GPUs would have trained on 2 and evaluated on 1."""
    job = _dry_run_spec(tmp_path, capsys, gpus=2, cpus=11)["job"]
    assert (job["resources"]["gpus"], job["resources"]["cpus"]) == (2, 11)


def test_an_empty_library_path_does_not_put_the_cwd_on_the_search_path(tmp_path, capsys):
    """`:${LIBRARY_PATH:-}` appends a trailing colon when the variable is unset, and an empty
    entry in a search path means the CURRENT DIRECTORY. gcc searches that path for a file named
    `specs`; the job runs with cwd=run_dir, where dataos-launch kept a DIRECTORY of that name —
    so every triton/inductor compile died with "cannot read spec file './specs': Is a directory"
    and took the vLLM engine with it. Three eval jobs lost on 2026-08-14, and the experimenter
    wrote its own workaround (moving the spec files aside between submits) rather than report it.
    """
    pre = "\n".join(_dry_run_spec(tmp_path, capsys)["job"]["pre_cmds"])
    assert "LIBRARY_PATH" in pre
    assert ':${LIBRARY_PATH:-}"' not in pre          # the trailing-colon form
    assert '${LIBRARY_PATH:+:$LIBRARY_PATH}"' in pre  # empty stays empty
