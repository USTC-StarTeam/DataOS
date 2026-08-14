"""dataos-eval CLI: submit / status / collect for the frozen PostTrainBench evaluator.

Runs ON the H100 login node (agents call it over ssh). The sbatch header is written by
THIS tool from env.yaml facts — agents never see a schedulable knob. Judge benches (declared in the task's eval/protocol.yaml) generate on the GPU node and
are scored at collect time on the login node — the only place with internet.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from dataos_launch import Handle, JobSpec, Resources, SlurmExtras, Status, get_launcher

_HANDLE = "eval_handle.json"


def _launcher_name(env: dict) -> str:
    """Which launcher runs this evaluation — declared by the environment, never guessed.

    Defaults to ``slurm`` because every environment that predates this field is a Slurm
    cluster; a workstation environment says ``launcher: local`` and the same JobSpec then runs
    as a subprocess. This is the whole point of the split: the evaluation protocol does not
    change with the machine, only the thing that starts it does.
    """
    return str((env.get("slurm") or {}).get("launcher") or env.get("launcher") or "slurm")


def _load(repo: Path, environment: str, task: str) -> tuple[dict, dict, dict]:
    env = yaml.safe_load(
        (repo / "configs" / "environments" / environment / "env.yaml").read_text()
    )
    task_cfg = yaml.safe_load(
        (repo / "configs" / "tasks" / task / "task.yaml").read_text()
    )
    protocol = yaml.safe_load(
        (repo / "configs" / "tasks" / task / "eval" / "protocol.yaml").read_text()
    )
    return env, task_cfg, protocol


def resolve_eval_template(repo: Path, task_config: str, bench: str, model_path: str,
                          protocol: dict) -> Path:
    """The protocol chat template for this model, from the MODEL's own config.json.

    Never from ``--model-name``/``--model-path`` substrings: that label is filled by the
    experimenter (task.yaml's ``ctx.label``), so guessing from it let an agent change the eval
    protocol by choosing a name. Keyed on ``model_type`` because a continuation node's model
    path is the parent's ``final_model`` dir and carries no family name, while a saved model
    always keeps its ``model_type``. Raises rather than guessing — a wrong chat template caps
    every score with nothing in any log.
    """
    cfg = Path(model_path) / "config.json"
    try:
        model_type = json.loads(cfg.read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {cfg} to resolve the chat template: {exc}") from exc
    family = (protocol.get("model_families") or {}).get(str(model_type))
    if not family:
        raise SystemExit(
            f"no chat template declared for model_type {model_type!r}; add it to "
            f"configs/tasks/{task_config}/eval/protocol.yaml model_families "
            "(together with the matching train_template)"
        )
    name = family.get(f"eval_template_{bench}") or family.get("eval_template")
    path = repo / "configs" / "tasks" / task_config / "eval" / "lmeval" / "templates" / name
    if not path.is_file():
        raise SystemExit(f"declared chat template does not exist: {path}")
    return path


def _build_job(args, env: dict) -> "JobSpec":
    """Render this evaluation as a launcher-agnostic JobSpec.

    Every hard-won environment lesson still lives here (short TMPDIR, libcuda link, cache
    redirects, offline HF) — but as ``pre_cmds`` and ``env`` on the spec rather than as text
    baked into a bash template, so the SAME evaluation can be run by any launcher. Where it
    runs is :mod:`dataos_launch`'s decision, made from the environment's declared launcher.

    Two things are deliberately explicit:

    * ``PYTORCH_CUDA_ALLOC_CONF`` is set to the empty string. ``build_preamble`` injects
      ``expandable_segments:True`` for TRAINING jobs, where fragmentation costs OOMs — but on
      an eval job it changes how vLLM's reservation is accounted (audit 12: a nominal 26 GiB
      was reported as 24320 MiB), and evaluation must not silently acquire a training default.
    * ``TMPDIR`` uses ``$SLURM_JOB_ID`` when Slurm sets it and the pid otherwise, because the
      path must stay short (the AF_UNIX socket limit is 108 chars) and unique per job on both
      a shared cluster node and a workstation.
    """
    import shlex

    run_root = shlex.quote(args.run_root)
    eval_py = f"{env['tools']['eval_env']['path']}/bin/python"
    repo = args.repo  # already absolute — resolved by the argument parser
    bridge = repo / "configs" / "tasks" / args.task_config / "eval" / "scripts"
    benchmarks = repo / "configs" / "tasks" / args.task_config / "benchmarks"
    slurm = env.get("slurm") or {}

    pre_cmds = [
        'export TMPDIR=/tmp/job_${SLURM_JOB_ID:-$$}',
        f'mkdir -p "$TMPDIR" "$TMPDIR/culib" {run_root}/cache/torch {run_root}/cache/triton '
        f'{run_root}/cache/inductor',
        "trap 'rm -rf \"$TMPDIR\"' EXIT",
        # triton/inductor compile needs libcuda.so; some nodes ship only libcuda.so.1
        "if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then",
        '  ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$TMPDIR/culib/libcuda.so"',
        "elif [ -e /usr/lib64/libcuda.so.1 ]; then",
        '  ln -sf /usr/lib64/libcuda.so.1 "$TMPDIR/culib/libcuda.so"',
        "fi",
        'export LIBRARY_PATH="$TMPDIR/culib:${LIBRARY_PATH:-}"',
        'export TRITON_LIBCUDA_PATH="$TMPDIR/culib"',
    ]
    job_env = {
        "TORCH_HOME": f"{args.run_root}/cache/torch",
        "TRITON_CACHE_DIR": f"{args.run_root}/cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{args.run_root}/cache/inductor",
        "POSTTRAINBENCH_BENCHMARKS_DIR": str(benchmarks),
        "HF_DATASETS_CACHE": f"{benchmarks}/hf_datasets",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "",  # see docstring: a training default, not an eval one
        # Server benches start a vLLM server and poll ITS OWN /health on 127.0.0.1. With an
        # HTTP proxy in the environment and no NO_PROXY, that request goes to the proxy, which
        # cannot reach a port on this machine — so the server comes up healthy and the poller
        # still times out after 10 minutes, blaming vLLM. No proxy is set on the cluster, which
        # is why this only ever appeared on a workstation.
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    return JobSpec(
        backend="lmeval",
        job_name=f"eval_{args.task}",
        command=[
            eval_py, str(bridge / "run_lmeval_eval.py"),
            "--task", args.task,
            "--model-path", args.model_path,
            "--model-name", args.model_name,
            "--chat-template", str(args.chat_template),
            "--output-dir", args.output_dir,
            "--limit", str(args.limit),
        ],
        env=job_env,
        run_dir=args.run_root,
        resources=Resources(gpus=args.gpus, cpus=args.cpus),
        pre_cmds=pre_cmds,
        slurm=SlurmExtras(
            partition=slurm.get("partition"),
            # The run chooses; env.yaml only declares what exists. Both of these read env-only
            # until 2026-08-14, and `env.slurm.nodes` is not a pin but a POOL — a run pinned to
            # gpu02 put 35 of 35 training jobs there and 68 of 94 eval jobs somewhere else,
            # because this was the one submitter that never saw the pin. `qos` was worse: when it
            # moved to the run config the env field vanished, jobs went out with no --qos at all,
            # and Slurm applied the account default (the GUARDED queue) in silence.
            qos=args.qos or slurm.get("qos") or (slurm.get("qos_available") or [None])[0],
            nodelist=args.nodelist or slurm.get("nodes"),
            cpus_per_task=args.cpus,
            extra_sbatch=[f"--time={_clamped_time(args, env)}"],
        ),
    )


def _time_s(v: str) -> int:
    """Slurm time -> seconds: [[days-]hours:]minutes[:seconds]; a bare number is minutes."""
    days, _, rest = v.partition("-")
    if not rest:
        days, rest = "0", v
    if ":" not in rest:
        return int(days) * 86400 + int(rest) * 60
    secs = 0
    for part in rest.split(":"):
        secs = secs * 60 + int(part)
    return int(days) * 86400 + secs


def _clamped_time(args, env: dict) -> str:
    """The job's ``--time``, clamped to the partition maximum.

    Explicit by necessity: with no ``--time`` sbatch falls back to the QOS's 7d limit, which
    EXCEEDS the partition's 3d MaxTime, and the job then pends forever with
    ``PartitionTimeLimit`` rather than failing.
    """
    limit = args.time
    max_time = (env.get("slurm") or {}).get("max_time")
    if max_time and _time_s(limit) > _time_s(max_time):
        print(f"dataos-eval: --time {limit} exceeds the partition max {max_time}; clamped",
              file=sys.stderr)
        limit = max_time
    return limit


def cmd_submit(args) -> int:
    repo = args.repo
    env, _, protocol = _load(repo, args.environment, args.task_config)
    bench_cfg = (protocol.get("benches") or {}).get(args.task)
    if bench_cfg is None:
        raise SystemExit(
            f"bench {args.task!r} is not declared in configs/tasks/{args.task_config}/eval/"
            "protocol.yaml benches"
        )
    # Resolved HERE, by the tool, from config + the model itself — not by the evaluator from
    # a caller-supplied label.
    args.chat_template = resolve_eval_template(
        repo, args.task_config, args.task, args.model_path, protocol
    )
    job = _build_job(args, env)
    name = _launcher_name(env)

    if args.dry_run:
        print(json.dumps({"launcher": name, "job": job.model_dump(mode="json")}, indent=1))
        return 0

    try:
        handle = get_launcher(name, **(env.get("access") or {})).submit(job)
    except Exception as exc:  # noqa: BLE001 - surface the launcher's own message
        print(f"submit failed via launcher {name!r}: {exc}", file=sys.stderr)
        return 1

    # The launch handle is the launcher's; this file adds what only EVALUATION knows —
    # which bench, where its outputs land, whether a judge still has to run.
    record = {
        "launch": handle.model_dump(mode="json"),
        "job_id": handle.job_id or str(handle.pid),   # kept for callers that read job_id
        "task": args.task,
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "model_name": args.model_name,
        "needs_judge": bool(bench_cfg.get("judge")),
        "chat_template": str(args.chat_template),
        "submitted_at": handle.submitted_at,
    }
    run_root = Path(args.run_root)
    (run_root / _HANDLE).write_text(json.dumps(record, indent=1))
    print(json.dumps({"job_id": record["job_id"], "handle": str(run_root / _HANDLE)}))
    return 0


def cmd_status(args) -> int:
    """Ask the launcher. The sacct polling and the state mapping that used to live here were a
    second copy of dataos-train's — its own docstring said "same policy as dataos-train's
    map_sacct_state", which is the sentence that precedes every drift."""
    record = json.loads(Path(args.handle).read_text())
    env, _, _ = _load(args.repo, args.environment, args.task_config)
    handle = Handle.model_validate(record["launch"])
    state = get_launcher(_launcher_name(env), **(env.get("access") or {})).status(handle)
    print(json.dumps({"job_id": record.get("job_id"), "state": str(state)}))
    return 0 if state in (Status.COMPLETED, Status.FAILED, Status.TIMEOUT) else 2


def cmd_collect(args) -> int:
    repo = args.repo
    env, task_cfg, _protocol = _load(repo, args.environment, args.task_config)
    handle = json.loads(Path(args.handle).read_text())
    summary_path = Path(handle["output_dir"]) / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    if handle.get("needs_judge") and summary.get("score") is None:
        # judge on the login node (the only place with internet) — bounded by design
        import shlex

        judge = (task_cfg.get("scoring") or {}).get("judge") or {}
        gateway = env.get("gateway") or {}
        bridge = args.repo / "configs" / "tasks" / args.task_config / "eval" / "scripts"
        benchmarks = args.repo / "configs" / "tasks" / args.task_config / "benchmarks"
        env_vars = (
            f"OPENAI_API_KEY=$(cat {shlex.quote(gateway['key_file'])}) "
            f"OPENAI_BASE_URL={shlex.quote(gateway['base_url'])} "
            f"POSTTRAINBENCH_JUDGE_MODEL={shlex.quote(str(judge.get('model')))} "
            f"POSTTRAINBENCH_BENCHMARKS_DIR={shlex.quote(str(benchmarks))}"
        )
        cmd = (
            f"{env_vars} {env['tools']['eval_env']['path']}/bin/python "
            f"{bridge}/run_lmeval_judge.py --bench {shlex.quote(handle['task'])} "
            f"--predictions {shlex.quote(handle['output_dir'])}/predictions.jsonl "
            f"--output {shlex.quote(str(summary_path))} --workers {judge.get('workers', 32)}"
        )
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            print(f"judge failed: {r.stderr.strip()[:500]}", file=sys.stderr)
            return 1
        summary = json.loads(summary_path.read_text())

    score = summary.get("score")
    print(json.dumps({"score": score, "summary": str(summary_path)}))
    return 0 if isinstance(score, (int, float)) else 1


def main() -> None:
    ap = argparse.ArgumentParser(prog="dataos-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        # Resolved at the boundary: every path this tool bakes into a job must be absolute,
        # because the job runs with cwd=run_dir, not cwd=repo. A relative --repo used to survive
        # on Slurm only because callers there happened to pass the deploy root absolute.
        p.add_argument("--repo", required=True, type=lambda v: Path(v).resolve(),
                       help="deploy checkout root (configs/ lives there)")
        p.add_argument("--environment", default="PTB_H100_slurm")
        p.add_argument("--task-config", default="posttrainbench")

    s = sub.add_parser("submit")
    common(s)
    s.add_argument("--task", required=True)
    s.add_argument("--model-path", required=True)
    s.add_argument("--model-name", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--run-root", required=True)
    s.add_argument("--limit", type=int, default=-1)
    # The run's resource facts. Defaults are a fallback for a caller that has none, NOT the
    # normal path — the caller has them and passes them (configs/tasks/*/task.yaml scoring.command).
    s.add_argument("--gpus", type=int, default=1)
    s.add_argument("--cpus", type=int, default=8)
    s.add_argument("--nodelist", default=None,
                   help="run-level node pin; overrides the environment's node pool")
    s.add_argument("--qos", default=None,
                   help="run-level QOS; env.yaml only declares which ones exist")
    s.add_argument("--time", default="12:00:00",
                   help="job time limit — must stay under the partition MaxTime (3d); "
                        "12h covers every full PTB eval with headroom")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_submit)

    for name, fn in (("status", cmd_status), ("collect", cmd_collect)):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("handle")
        p.set_defaults(fn=fn)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
