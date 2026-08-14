"""``dataos-train`` CLI — JSON in / JSON out; the deterministic surface an agent drives.

Lifecycle: ``build`` (TrainSpec->Job) · ``submit`` (->Handle, returns now) · ``status`` ·
``logs`` · ``collect`` (->TrainResult) · ``run`` (blocking submit+poll+collect) · ``capabilities``
· ``schema``. Model jobs (score data with a model, no training) mirror the same shape:
``modeljob-validate`` / ``modeljob-build`` / ``modeljob-collect``.
Each command prints exactly one JSON object on stdout and also persists its durable artifact
(spec.json / handle.json / result.json / score_result.json) into ``run_dir`` so a resumed
agent reads from disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from dataos_launch.io import atomic_write_json, read_json
from dataos_launch.launchers.base import materialize
from dataos_launch.placement import launcher_from_config
from dataos_launch.registry import get_launcher
from dataos_launch.spec import SpecError
from dataos_launch.types import Handle, Job, Status

from .modeljob import ModelJobSpec
from .modeljob import backend as modeljob_backend
from .registry import backend_names, get_backend
from .schemas import (
    all_schemas,
    backend_method_schema,
    backend_schema,
    method_schema,
    model_schema,
    schema_names,
)
from .precheck import precheck_disk
from .spec import TrainSpec
from .stage import TrainStage


def _read_input(arg: str) -> Any:
    if arg == "-":
        return json.loads(sys.stdin.read())
    return read_json(arg)


def _emit(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, default=str))


def _persist_job(job: Job) -> None:
    Path(job.run_dir).mkdir(parents=True, exist_ok=True)
    # `job.json`, not `spec.json`. The agent authors a TrainSpec and this used to overwrite it
    # with the rendered Job at the same path, so "the spec" meant two different documents
    # depending on whether build had run — and re-submitting the same path submitted the Job.
    atomic_write_json(Path(job.run_dir) / "job.json", job.model_dump())
    materialize(job)  # write rendered assets (run_train.sh, configs) into run_dir
    # (build IS the render step)


def _stage(args: argparse.Namespace) -> TrainStage:
    """The stage, built from the deployment — the same way dataos-eval and dataos-data do it.

    ``--repo``/``--environment`` replace ``--launcher`` and ``--cluster-json``. The environment
    declares which launcher runs here and what the partition's limits are; the run's own choices
    (``--qos``, ``--nodelist``) go to the launcher beside them. Naming a launcher on the command
    line was a fourth place placement could be decided, and the one furthest from the config.
    """
    return TrainStage.from_repo(args.repo, args.environment)


def _resource(args: argparse.Namespace) -> dict:
    return {"qos": getattr(args, "qos", None), "node": getattr(args, "nodelist", None),
            "partition": getattr(args, "partition", None)}


def _looks_like_jobspec(data: dict) -> bool:
    return all(k in data for k in ("command", "run_dir", "backend", "job_name"))


def _build_or_die(stage: TrainStage, spec: TrainSpec) -> Job:
    """Validate (schema + disk) then build. The funnel both ``build`` and ``submit`` go through:
    submit used to skip validation entirely, so a spec the backend rejected at build time was
    still submitted when the caller passed a TrainSpec straight to ``submit``."""
    try:
        return stage.prepare(spec)
    except SpecError as exc:
        _emit({"ok": False, "errors": exc.errors})
        raise SystemExit(1) from None


def _load_jobspec_or_build(stage: TrainStage, arg: str) -> Job:
    """The Job to run, and the placement its spec carried.

    Accepts either document: a TrainSpec (validated and built here) or an already-built Job
    from ``job.json``, for a caller that inspected or hand-patched the render before going.
    """
    data = _read_input(arg)
    if isinstance(data, dict) and _looks_like_jobspec(data):
        return Job.model_validate(data)
    return _build_or_die(stage, TrainSpec.model_validate(data))


def _backend_for_run(run_dir: str, fallback: str) -> str:
    spec_path = Path(run_dir) / "job.json"
    if spec_path.is_file():
        try:
            return read_json(spec_path).get("backend", fallback)
        except (OSError, json.JSONDecodeError, ValueError):
            return fallback
    return fallback


def _finalize(handle: Handle, launcher: Any, status: Status, fallback_backend: str) -> int:
    backend = get_backend(_backend_for_run(handle.run_dir, fallback_backend))
    result = backend.collect(handle.run_dir)
    result.spec_id = handle.spec_id
    if result.status == Status.COMPLETED and status != Status.COMPLETED:
        result.status = status  # outputs may exist but the job didn't finish cleanly
    if not result.error and result.status != Status.COMPLETED:
        result.error = f"job ended with status {status.value}"
    if not result.log_tail:
        result.log_tail = launcher.logs(handle, tail=50)
    atomic_write_json(Path(handle.run_dir) / "result.json", result.model_dump())
    _emit(result.model_dump())
    return 0 if result.status == Status.COMPLETED else 1


def cmd_capabilities(args: argparse.Namespace) -> int:
    if args.backend:
        _emit(get_backend(args.backend).capabilities().model_dump())
    else:
        out: dict[str, Any] = {}
        for name in backend_names():
            try:
                out[name] = get_backend(name).capabilities().model_dump()
            except Exception as exc:  # noqa: BLE001
                out[name] = {"error": str(exc)}
        _emit(out)
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    if args.backend and args.method:
        _emit(backend_method_schema(args.backend, args.method))
    elif args.backend:
        _emit(backend_schema(args.backend))
    elif args.method:
        _emit(method_schema(args.method))
    elif args.name in ("all", "*"):
        _emit(all_schemas())
    elif args.name:
        _emit(model_schema(args.name))
    else:
        _emit(all_schemas())
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    spec = TrainSpec.model_validate(_read_input(args.spec))
    # same checks build/submit apply, so `validate` cannot report ok on a spec they will reject
    errors = get_backend(spec.backend).validate(spec) + precheck_disk(spec)
    _emit({"ok": not errors, "errors": errors})
    return 0 if not errors else 1


def cmd_build(args: argparse.Namespace) -> int:
    spec = TrainSpec.model_validate(_read_input(args.spec))
    try:
        job = _build_or_die(_stage(args), spec)
    except SystemExit:
        return 1
    _persist_job(job)
    _emit(job.model_dump())
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    stage = _stage(args)
    job = _load_jobspec_or_build(stage, args.spec)
    _persist_job(job)
    handle = launcher_from_config(stage.env, _resource(args)).submit(job)
    _emit(handle.model_dump())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    handle = Handle.model_validate(_read_input(args.handle))
    launcher = get_launcher(handle.launcher)
    st = launcher.status(handle)
    atomic_write_json(Path(handle.run_dir) / "handle.json", handle.model_dump())
    _emit({"status": st.value, "handle": handle.model_dump()})
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    handle = Handle.model_validate(_read_input(args.handle))
    get_launcher(handle.launcher).cancel(handle)
    _emit({"cancelled": handle.job_id})
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    handle = Handle.model_validate(_read_input(args.handle))
    launcher = get_launcher(handle.launcher)
    text = launcher.logs(handle, tail=args.tail)
    if args.raw:
        sys.stdout.write(text)
    else:
        _emit({"log": text, "path": handle.log_path})
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    if args.run_dir:
        # externally-submitted jobs (e.g. via an environment layer's own sbatch): no
        # handle.json exists — collect straight from the artifacts on disk.
        backend = get_backend(_backend_for_run(args.run_dir, args.backend))
        result = backend.collect(args.run_dir)
        atomic_write_json(Path(args.run_dir) / "result.json", result.model_dump())
        _emit(result.model_dump())
        return 0 if result.status == Status.COMPLETED else 2
    handle = Handle.model_validate(_read_input(args.handle))
    launcher = get_launcher(handle.launcher)
    st = launcher.status(handle)
    if not st.is_terminal() and not args.force:
        # Collecting a live job used to fabricate a FAILED result (outputs aren't there
        # yet) — surface the live status instead; --force keeps the old behavior.
        _emit({"status": st.value, "error": "job is not terminal yet; poll status, then collect"})
        return 3
    return _finalize(handle, launcher, st, fallback_backend="dummy")


def cmd_run(args: argparse.Namespace) -> int:
    stage = _stage(args)
    job = _load_jobspec_or_build(stage, args.spec)
    _persist_job(job)
    launcher = launcher_from_config(stage.env, _resource(args))
    handle = launcher.submit(job)
    deadline = (time.monotonic() + args.timeout) if args.timeout else None
    st = launcher.status(handle)
    while not st.is_terminal():
        if deadline and time.monotonic() > deadline:
            launcher.cancel(handle)
            st = Status.TIMEOUT
            break
        time.sleep(args.poll)
        st = launcher.status(handle)
    return _finalize(handle, launcher, st, fallback_backend=job.backend)


def cmd_modeljob_validate(args: argparse.Namespace) -> int:
    spec = ModelJobSpec.model_validate(_read_input(args.spec))
    errors = modeljob_backend.validate(spec)
    _emit({"ok": not errors, "errors": errors})
    return 0 if not errors else 1


def cmd_modeljob_build(args: argparse.Namespace) -> int:
    spec = ModelJobSpec.model_validate(_read_input(args.spec))
    errors = modeljob_backend.validate(spec)
    if errors:
        _emit({"ok": False, "errors": errors})
        return 1
    job = modeljob_backend.build(spec, run_dir=args.run_dir)
    _persist_job(job)
    _emit(job.model_dump())
    return 0


def cmd_modeljob_collect(args: argparse.Namespace) -> int:
    result = modeljob_backend.collect(args.run_dir)
    rd = Path(args.run_dir)
    if rd.is_dir():
        atomic_write_json(rd / "score_result.json", result.model_dump())
    _emit(result.model_dump())
    return 0 if result.status == Status.COMPLETED else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dataos-train", description="Unified multi-backend training.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="TrainSpec -> Job")
    b.add_argument("spec", help="path to TrainSpec JSON, or - for stdin")
    b.add_argument("--repo", required=True,
                   help="deploy checkout root (configs/ lives there)")
    b.add_argument("--environment", default="PTB_H100_slurm")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="TrainSpec -> {ok, errors}")
    v.add_argument("spec")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("submit", help="Job/TrainSpec -> Handle (returns immediately)")
    s.add_argument("spec")
    s.set_defaults(func=cmd_submit)

    st = sub.add_parser("status", help="Handle -> status")
    st.add_argument("handle")
    st.set_defaults(func=cmd_status)

    cn = sub.add_parser("cancel", help="Handle -> stop the job it names")
    cn.add_argument("handle")
    cn.set_defaults(func=cmd_cancel)

    lg = sub.add_parser("logs", help="Handle -> logs")
    lg.add_argument("handle")
    lg.add_argument("--tail", type=int, default=None)
    lg.add_argument("--raw", action="store_true")
    lg.set_defaults(func=cmd_logs)

    c = sub.add_parser("collect", help="Handle -> TrainResult (refuses on a live job)")
    c.add_argument("handle", nargs="?", default=None)
    c.add_argument("--run-dir", dest="run_dir", default=None,
                   help="collect straight from a run_dir (job was submitted by an external "
                        "launcher; no handle.json)")
    c.add_argument("--backend", default="llamafactory",
                   help="backend for --run-dir mode when run_dir/spec.json is absent")
    c.add_argument("--force", action="store_true", help="finalize even if not terminal")
    c.set_defaults(func=cmd_collect)

    r = sub.add_parser("run", help="TrainSpec/Job -> TrainResult (blocking)")
    r.add_argument("spec")
    for sub_p in (s, r):  # NOT `p`: that name is the top-level parser this function returns
        sub_p.add_argument("--repo", required=True,
                           help="deploy checkout root (configs/ lives there)")
        sub_p.add_argument("--environment", default="PTB_H100_slurm")
        sub_p.add_argument("--qos", default=None,
                           help="run-level QOS; env.yaml only declares which ones exist")
        sub_p.add_argument("--nodelist", default=None,
                           help="run-level node pin; overrides the environment's node pool")
        sub_p.add_argument("--partition", default=None,
                           help="run-level partition; env.yaml names the default")
    r.add_argument("--poll", type=float, default=10.0)
    r.add_argument("--timeout", type=float, default=None)
    r.set_defaults(func=cmd_run)

    mjv = sub.add_parser("modeljob-validate", help="ModelJobSpec -> {ok, errors}")
    mjv.add_argument("spec", help="path to ModelJobSpec JSON, or - for stdin")
    mjv.set_defaults(func=cmd_modeljob_validate)

    mjb = sub.add_parser("modeljob-build", help="ModelJobSpec -> Job (renders run_score.sh)")
    mjb.add_argument("spec")
    mjb.add_argument(
        "--run-dir",
        dest="run_dir",
        default=None,
        help="job run dir (default: modeljobs/<name> next to the dataset)",
    )
    mjb.set_defaults(func=cmd_modeljob_build)

    mjc = sub.add_parser(
        "modeljob-collect", help="run_dir -> ScoreResult (parses scores.jsonl, "
        "writes score_result.json)"
    )
    mjc.add_argument("run_dir")
    mjc.set_defaults(func=cmd_modeljob_collect)

    cap = sub.add_parser("capabilities", help="backend capabilities")
    cap.add_argument("backend", nargs="?", default=None)
    cap.set_defaults(func=cmd_capabilities)

    sch = sub.add_parser("schema", help="machine-readable TrainSpec/method schemas")
    sch.add_argument(
        "name",
        nargs="?",
        default=None,
        help=f"named schema ({', '.join(schema_names())}); omit for full bundle",
    )
    sch.add_argument("--backend", default=None, help="include method schemas for a backend")
    sch.add_argument("--method", default=None, help="schema for a method such as sft or grpo")
    sch.set_defaults(func=cmd_schema)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyError as exc:
        _emit({"error": str(exc).strip('"')})
        return 2
    except Exception as exc:  # noqa: BLE001
        _emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
