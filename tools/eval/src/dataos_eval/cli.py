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
import time
from pathlib import Path

import yaml

_HANDLE = "eval_handle.json"


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


def _render_job(args, env: dict) -> str:
    """The full job script: every hard-won environment lesson baked in (short TMPDIR,
    libcuda link, cache redirects, offline HF). No scheduling flags exist here — those
    are passed to sbatch by submit() and are not the caller's business. All interpolated
    paths are shlex-quoted (run roots can contain parens, e.g. fresh-instance -1 dirs)."""
    import shlex

    run_root = shlex.quote(args.run_root)
    model_path = shlex.quote(args.model_path)
    model_name = shlex.quote(args.model_name)
    output_dir = shlex.quote(args.output_dir)
    bench = shlex.quote(args.task)
    chat_template = shlex.quote(str(args.chat_template))
    eval_py = f"{env['tools']['eval_env']['path']}/bin/python"
    bridge = Path(args.repo) / "configs" / "tasks" / args.task_config / "eval" / "scripts"
    benchmarks = Path(args.repo) / "configs" / "tasks" / args.task_config / "benchmarks"
    return f"""#!/bin/bash
set -euo pipefail
export TMPDIR=/tmp/job_${{SLURM_JOB_ID}}
mkdir -p "$TMPDIR" "$TMPDIR/culib" {run_root}/cache/torch {run_root}/cache/triton {run_root}/cache/inductor
trap 'rm -rf "$TMPDIR"' EXIT
export TORCH_HOME={run_root}/cache/torch
export TRITON_CACHE_DIR={run_root}/cache/triton
export TORCHINDUCTOR_CACHE_DIR={run_root}/cache/inductor
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
  ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$TMPDIR/culib/libcuda.so"
elif [ -e /usr/lib64/libcuda.so.1 ]; then
  ln -sf /usr/lib64/libcuda.so.1 "$TMPDIR/culib/libcuda.so"
fi
export LIBRARY_PATH="$TMPDIR/culib:${{LIBRARY_PATH:-}}"
export TRITON_LIBCUDA_PATH="$TMPDIR/culib"
export POSTTRAINBENCH_BENCHMARKS_DIR={benchmarks}
export HF_DATASETS_CACHE={benchmarks}/hf_datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
{eval_py} {bridge}/run_lmeval_eval.py \\
  --task {bench} \\
  --model-path {model_path} \\
  --model-name {model_name} \\
  --chat-template {chat_template} \\
  --output-dir {output_dir} \\
  --limit {args.limit}
"""


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


def _sbatch_argv(args, env: dict) -> list[str]:
    """Scheduling facts, written here and ONLY here: single node, env node pool,
    partition/qos from env.yaml, gres/cpus from the run's resource facts. --time is
    explicit: with none, sbatch falls back to the QOS's 7d limit, which EXCEEDS the
    partition's 3d MaxTime and pends forever (PartitionTimeLimit); anything above
    slurm.max_time is clamped to it for the same reason."""
    slurm = env["slurm"]
    limit = args.time
    max_time = slurm.get("max_time")
    if max_time and _time_s(limit) > _time_s(max_time):
        print(f"dataos-eval: --time {limit} exceeds the partition max {max_time}; clamped",
              file=sys.stderr)
        limit = max_time
    argv = [
        "sbatch", "--parsable",
        "-N1", "-n1",
        f"--partition={slurm['partition']}",
        f"--qos={slurm['qos']}",
        f"--gres=gpu:{args.gpus}",
        f"--cpus-per-task={args.cpus}",
        f"--time={limit}",
    ]
    if slurm.get("nodes"):
        argv.append(f"--nodelist={slurm['nodes']}")
    return argv


def cmd_submit(args) -> int:
    repo = Path(args.repo)
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
    run_root = Path(args.run_root)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    script = run_root / "eval_job.sh"
    script.write_text(_render_job(args, env))

    argv = _sbatch_argv(args, env) + [
        f"--chdir={run_root}",
        f"--output={run_root}/logs/slurm-%j.out",
        f"--error={run_root}/logs/slurm-%j.err",
        str(script),
    ]
    if args.dry_run:
        print(" ".join(argv))
        print(script.read_text())
        return 0
    out = subprocess.run(argv, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        print(f"sbatch failed: {out.stderr.strip()}", file=sys.stderr)
        return 1
    job_id = out.stdout.strip().split(";")[0]
    handle = {
        "job_id": job_id,
        "task": args.task,
        "output_dir": args.output_dir,
        "model_path": args.model_path,
        "model_name": args.model_name,
        "needs_judge": bool(bench_cfg.get("judge")),
        "chat_template": str(args.chat_template),
        "script": str(script),
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_root / _HANDLE).write_text(json.dumps(handle, indent=1))
    print(json.dumps({"job_id": job_id, "handle": str(run_root / _HANDLE)}))
    return 0


_FAIL = {"FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "BOOT_FAIL", "PREEMPTED", "REVOKED"}
_TMO = {"TIMEOUT", "DEADLINE"}


def _map_state(raw: str) -> str:
    """Coarse terminal/running mapping for one sacct State field — tolerates the reason
    suffix (`CANCELLED by 1003`) and the long tail of fail states. Same policy as
    dataos-train's map_sacct_state."""
    token = (raw or "").strip()
    state = token.split()[0].upper() if token else ""
    if state == "COMPLETED":
        return "COMPLETED"
    if state in _FAIL or state.startswith("CANCELLED"):
        return "FAILED"
    if state in _TMO:
        return "TIMEOUT"
    if state in ("RUNNING", "COMPLETING", "RESIZING", "SUSPENDED"):
        return "RUNNING"
    return "PENDING"


def _state(job_id: str) -> str:
    out = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "-n"],
        capture_output=True, text=True, check=False,
    )
    states = [s.strip() for s in out.stdout.splitlines() if s.strip()]
    # first non-empty row = the primary job, NOT .batch/.extern (a failed job's .extern
    # is often COMPLETED — reading the LAST row misreports FAILED evals as COMPLETED)
    return _map_state(states[0]) if states else "PENDING"


def cmd_status(args) -> int:
    handle = json.loads(Path(args.handle).read_text())
    state = _state(handle["job_id"])
    print(json.dumps({"job_id": handle["job_id"], "state": state}))
    return 0 if state in ("COMPLETED", "FAILED", "TIMEOUT") else 2


def cmd_collect(args) -> int:
    repo = Path(args.repo)
    env, task_cfg, _protocol = _load(repo, args.environment, args.task_config)
    handle = json.loads(Path(args.handle).read_text())
    summary_path = Path(handle["output_dir"]) / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    if handle.get("needs_judge") and summary.get("score") is None:
        # judge on the login node (the only place with internet) — bounded by design
        import shlex

        judge = (task_cfg.get("scoring") or {}).get("judge") or {}
        gateway = env.get("gateway") or {}
        bridge = Path(args.repo) / "configs" / "tasks" / args.task_config / "eval" / "scripts"
        benchmarks = Path(args.repo) / "configs" / "tasks" / args.task_config / "benchmarks"
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
        p.add_argument("--repo", required=True, help="deploy checkout root (configs/ lives there)")
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
    s.add_argument("--gpus", type=int, default=1)
    s.add_argument("--cpus", type=int, default=8)
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
