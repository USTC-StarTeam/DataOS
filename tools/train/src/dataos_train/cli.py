"""``dataos-train`` CLI — JSON in / JSON out; the deterministic surface an agent drives.

Lifecycle: ``build`` (TrainSpec->JobSpec) · ``submit`` (->Handle, returns now) · ``status`` ·
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
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from .io import atomic_write_json, read_json
from .launchers.base import materialize
from .modeljob import ModelJobSpec
from .modeljob import backend as modeljob_backend
from .registry import backend_names, get_backend, get_launcher
from .schemas import (
    all_schemas,
    backend_method_schema,
    backend_schema,
    method_schema,
    model_schema,
    schema_names,
)
from .spec import TrainSpec
from .types import Handle, JobSpec, Status


def _read_input(arg: str) -> Any:
    if arg == "-":
        return json.loads(sys.stdin.read())
    return read_json(arg)


def _emit(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, default=str))


def _persist_job(job: JobSpec) -> None:
    Path(job.run_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_json(Path(job.run_dir) / "spec.json", job.model_dump())
    materialize(job)  # write rendered assets (run_train.sh, configs) into run_dir
    # (build IS the render step)


#: Free space required before a run is allowed to start, as a multiple of the base model's
#: weight files. A full-SFT run writes back roughly one model (``final_model``); a LoRA run writes
#: an adapter plus a merged export. The headroom covers the export and a little slack. It is
#: still a FLOOR, not a guarantee: nothing here reserves the space against another job filling
#: the same filesystem in the meantime.
_DISK_HEADROOM = 1.5


def _retained_models(spec: TrainSpec) -> float:
    """How many model-sized artifacts this run keeps on disk.

    Trajectory checkpoints (``save_steps``) each cost one model's worth — ``save_only_model`` is
    forced in the rendered config, so no optimizer state — but only for full/freeze tuning. A
    LoRA checkpoint is an adapter, orders of magnitude smaller, so it does not move this number.
    """
    method = spec.method
    if not getattr(method, "save_steps", None):
        return _DISK_HEADROOM
    if getattr(method, "finetuning_type", None) == "lora":
        return _DISK_HEADROOM
    return _DISK_HEADROOM + int(getattr(method, "save_total_limit", 0) or 0)


def _model_bytes(path: str) -> int:
    """Size of the model's weight files, or 0 when the path is not a readable model dir."""
    root = Path(path)
    if not root.is_dir():
        return 0
    return sum(
        f.stat().st_size
        for f in root.iterdir()
        if f.is_file() and f.suffix in {".safetensors", ".bin", ".pt"}
    )


def precheck_disk(spec: TrainSpec) -> list[str]:
    """Refuse a run that cannot fit its own output.

    The 4b2 v5 batch exhausted ``/data2`` mid-run: training finished, then ``save_pretrained``
    died writing the weights, so the GPU hours were spent and nothing came back. The signature is
    unhelpful too — a full filesystem surfaces as safetensors' "File too large (os error 27)",
    which reads like a size bug rather than a space one. Checking before submit turns hours of
    wasted compute into one line of output.
    """
    factor = _retained_models(spec)
    need = int(_model_bytes(spec.base_model) * factor)
    if not need:  # unreadable base (e.g. a spec built off-cluster) — nothing to compare against
        return []
    probe = Path(spec.run_dir)
    while not probe.exists() and probe != probe.parent:  # run_dir is created by the job
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free >= need:
        return []
    gb = 1024**3
    keep = getattr(spec.method, "save_steps", None) and getattr(spec.method, "save_total_limit", 0)
    return [
        f"not enough free space under {spec.run_dir}: {free / gb:.1f} GiB free, this run needs "
        f"at least {need / gb:.1f} GiB ({factor}x the base model's "
        f"{_model_bytes(spec.base_model) / gb:.1f} GiB of weights) to write its output"
        + (f", including {keep} retained checkpoints" if keep else "")
        + ". Free space, lower save_total_limit, or point run_dir at another filesystem — "
        "training to completion and then failing on save wastes the whole allocation."
    ]


def _looks_like_jobspec(data: dict) -> bool:
    return all(k in data for k in ("command", "run_dir", "backend", "job_name"))


def _build_or_die(spec: TrainSpec) -> JobSpec:
    """Validate (schema + disk) then build. The single funnel for ``build`` and ``submit``:
    submit used to skip validation entirely, so a spec the backend rejected at build time was
    still submitted when the caller passed a TrainSpec straight to ``submit``."""
    backend = get_backend(spec.backend)
    errors = backend.validate(spec) + precheck_disk(spec)
    if errors:
        _emit({"ok": False, "errors": errors})
        raise SystemExit(1)
    return backend.build(spec)


def _load_jobspec_or_build(arg: str) -> JobSpec:
    data = _read_input(arg)
    if isinstance(data, dict) and _looks_like_jobspec(data):
        return JobSpec.model_validate(data)
    return _build_or_die(TrainSpec.model_validate(data))


def _backend_for_run(run_dir: str, fallback: str) -> str:
    spec_path = Path(run_dir) / "spec.json"
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
        job = _build_or_die(spec)
    except SystemExit:
        return 1
    _persist_job(job)
    _emit(job.model_dump())
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    job = _load_jobspec_or_build(args.spec)
    _persist_job(job)
    cluster = _read_input(args.cluster_json) if args.cluster_json else {}
    launcher = get_launcher(args.launcher, **cluster)
    handle = launcher.submit(job)
    _emit(handle.model_dump())
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    handle = Handle.model_validate(_read_input(args.handle))
    launcher = get_launcher(handle.launcher)
    st = launcher.status(handle)
    atomic_write_json(Path(handle.run_dir) / "handle.json", handle.model_dump())
    _emit({"status": st.value, "handle": handle.model_dump()})
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    handle = Handle.model_validate(_read_input(args.handle))
    launcher = get_launcher(handle.launcher)
    text = launcher.logs(handle, tail=args.tail)
    if args.raw:
        sys.stdout.write(text)
    else:
        _emit({"log": text, "path": handle.log_path or handle.stdout_path})
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
    job = _load_jobspec_or_build(args.spec)
    _persist_job(job)
    cluster = _read_input(args.cluster_json) if args.cluster_json else {}
    launcher = get_launcher(args.launcher, **cluster)
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

    b = sub.add_parser("build", help="TrainSpec -> JobSpec")
    b.add_argument("spec", help="path to TrainSpec JSON, or - for stdin")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="TrainSpec -> {ok, errors}")
    v.add_argument("spec")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("submit", help="JobSpec/TrainSpec -> Handle (returns immediately)")
    s.add_argument("spec")
    s.add_argument("--launcher", default="local", choices=["local", "slurm"])
    s.add_argument("--cluster-json", dest="cluster_json", default=None)
    s.set_defaults(func=cmd_submit)

    st = sub.add_parser("status", help="Handle -> status")
    st.add_argument("handle")
    st.set_defaults(func=cmd_status)

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

    r = sub.add_parser("run", help="TrainSpec/JobSpec -> TrainResult (blocking)")
    r.add_argument("spec")
    r.add_argument("--launcher", default="local", choices=["local", "slurm"])
    r.add_argument("--cluster-json", dest="cluster_json", default=None)
    r.add_argument("--poll", type=float, default=10.0)
    r.add_argument("--timeout", type=float, default=None)
    r.set_defaults(func=cmd_run)

    mjv = sub.add_parser("modeljob-validate", help="ModelJobSpec -> {ok, errors}")
    mjv.add_argument("spec", help="path to ModelJobSpec JSON, or - for stdin")
    mjv.set_defaults(func=cmd_modeljob_validate)

    mjb = sub.add_parser("modeljob-build", help="ModelJobSpec -> JobSpec (renders run_score.sh)")
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
