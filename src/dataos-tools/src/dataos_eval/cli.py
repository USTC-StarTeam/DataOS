"""dataos-eval CLI: submit / status / collect for the frozen PostTrainBench evaluator.

Runs ON the H100 login node (agents call it over ssh). Agents never see a schedulable knob: what
an evaluation IS lives in :mod:`dataos_eval.stage`, where it runs is the launcher's, and this
file only parses flags and prints. Judge benches (declared in the task's ``eval/protocol.yaml``)
generate on the GPU node and are scored at collect time on the login node — the only place with
internet.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dataos_launch import Handle, Resources
from dataos_launch.launchers.slurm import slurm_time_s
from dataos_launch.spec import SpecError

from .stage import EvalStage

#: Suffix, not a name: the spec's own name goes in front so a checkpoint sweep's six
#: evaluations leave six records instead of overwriting one. `collect` and `status`
#: take the path this prints, so nothing has to guess it.
_HANDLE = "eval_handle.json"


def _stage(args) -> EvalStage:
    return EvalStage.from_repo(args.repo, args.environment, args.task_config)


def _spec(args, stage: EvalStage):
    return stage.spec_for(
        task=args.task,
        model_path=args.model_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        run_root=args.run_root,
        limit=args.limit,
        resources=Resources(
            gpus=args.gpus, cpus=args.cpus, time_h=slurm_time_s(args.time) / 3600
        ),
    )


def cmd_submit(args) -> int:
    stage = _stage(args)
    spec = _spec(args, stage)
    # The run config's `resource` block, arriving as flags because the agent calls this over ssh.
    resource = {"qos": args.qos, "node": args.nodelist, "partition": args.partition}

    if args.dry_run:
        print(json.dumps(stage.preview(spec, resource), indent=1))
        return 0

    try:
        handle = stage.submit(spec, resource)
    except SpecError as exc:
        print(json.dumps({"ok": False, "errors": exc.errors}), file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface the launcher's own message
        print(f"submit failed: {exc}", file=sys.stderr)
        return 1

    # The launch handle is the launcher's; the rest is what only EVALUATION knows — which bench,
    # where its outputs land, whether a judge still has to run. Both come off the spec, so the
    # record cannot disagree with what was submitted.
    record = {
        "launch": handle.model_dump(mode="json"),
        "job_id": handle.job_id,
        "task": spec.task,
        "output_dir": spec.output_dir,
        "model_path": spec.model_path,
        "model_name": spec.model_name,
        "needs_judge": spec.needs_judge,
        "chat_template": spec.chat_template,
        "submitted_at": handle.submitted_at,
    }
    spec.write()
    path = Path(spec.run_dir) / f"{spec.name}.{_HANDLE}"
    path.write_text(json.dumps(record, indent=1))
    print(json.dumps({"job_id": record["job_id"], "handle": str(path)}))
    return 0


def cmd_status(args) -> int:
    """Ask the launcher. Polling needs no placement — the handle says who took the job, which
    is also why this no longer reads env.yaml (and no longer wanted flags its parser lacked)."""
    record = json.loads(Path(args.handle).read_text())
    state = EvalStage.status(Handle.model_validate(record["launch"]))
    print(json.dumps({"job_id": record.get("job_id"), "state": str(state)}))
    return 0 if state.is_terminal() else 2


def cmd_cancel(args) -> int:
    record = json.loads(Path(args.handle).read_text())
    handle = Handle.model_validate(record["launch"])
    EvalStage.cancel(handle)
    print(json.dumps({"cancelled": handle.job_id}))
    return 0


def cmd_collect(args) -> int:
    stage = _stage(args)
    record = json.loads(Path(args.handle).read_text())
    summary_path = Path(record["output_dir"]) / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    if record.get("needs_judge") and summary.get("score") is None:
        cmd = stage.judge_command(record["task"], record["output_dir"], summary_path)
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
        p.add_argument("--environment", default="example-slurm")
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
    s.add_argument("--partition", default=None,
                   help="run-level partition; env.yaml names the default")
    s.add_argument("--time", default="12:00:00",
                   help="job time limit — must stay under the partition MaxTime (3d); "
                        "12h covers every full PTB eval with headroom")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_submit)

    p = sub.add_parser("status")
    p.add_argument("handle")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("cancel")
    p.add_argument("handle")
    p.set_defaults(fn=cmd_cancel)

    p = sub.add_parser("collect")
    common(p)
    p.add_argument("handle")
    p.set_defaults(fn=cmd_collect)

    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
