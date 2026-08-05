"""Shared pipeline harness for real DataOS-train GPU validation runs.

Pipeline scripts define the experiment facts. This module performs the runtime-owned
work: data/model preparation, TrainSpec construction, backend validation/build,
local launch, collection, manifest writing, and failure diagnosis.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dataos_train.io import atomic_write_json, atomic_write_text, read_json, utc_now_iso
from dataos_train.launchers.base import materialize
from dataos_train.registry import get_backend, get_launcher
from dataos_train.spec import DatasetRef, EnvSpec, TrainSpec
from dataos_train.types import Handle, JobSpec, Resources, Status, TrainResult

MethodFactory = Callable[["PipelineContext"], Any]

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CONDA_SH = "/home/hcgu/miniconda3/etc/profile.d/conda.sh"
DEFAULT_ENV_ROOT = "/home/hcgu/project/DataOS/.conda/envs"
DEFAULT_LF_ENV = f"{DEFAULT_ENV_ROOT}/dataos-train-lf-cu121-py311"
DEFAULT_LF_PPO_ENV = f"{DEFAULT_ENV_ROOT}/dataos-train-lf-ppo-cu121-py311"
DEFAULT_VERL_ENV = f"{DEFAULT_ENV_ROOT}/dataos-train-verl-cu121-py311"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


class PipelineBlocked(RuntimeError):
    """The pipeline cannot run until an external dependency is prepared."""


@dataclass(frozen=True)
class DatasetPlan:
    kind: str
    source: str = ""
    split: str = "train"
    format: str = "auto"
    sample_size: int = 64
    output_name: str = "source.jsonl"
    repeat_to: int | None = None
    records: tuple[dict[str, Any], ...] = ()
    raw_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelPlan:
    env_var: str = "DATAOS_TRAIN_BASE_MODEL"
    default_ref: str = DEFAULT_BASE_MODEL
    local_subdir: str = "qwen2.5-0.5b-instruct"


@dataclass(frozen=True)
class PipelineDefinition:
    name: str
    backend: str
    dataset: DatasetPlan
    method_factory: MethodFactory
    resources: Resources
    model: ModelPlan = field(default_factory=ModelPlan)
    expected_failure: bool = False
    expected_failure_reason: str = ""
    acceptance_artifacts: tuple[str, ...] = ("final_model",)
    env_extra: dict[str, str] = field(default_factory=dict)
    backend_env: str | None = None


@dataclass
class PipelineContext:
    definition: PipelineDefinition
    args: argparse.Namespace
    repo_root: Path
    workspace_root: Path
    runs_root: Path
    run_dir: Path
    cache_root: Path
    manifest: dict[str, Any]

    def require_reward_model(self) -> str:
        reward_model = self.args.reward_model or os.environ.get("DATAOS_TRAIN_REWARD_MODEL")
        if not reward_model:
            raise PipelineBlocked(
                "reward model is required; pass --reward-model or set DATAOS_TRAIN_REWARD_MODEL"
            )
        return reward_model


def constant_method(method: Any) -> MethodFactory:
    def _factory(_: PipelineContext) -> Any:
        return method.model_copy(deep=True) if hasattr(method, "model_copy") else method

    return _factory


def run_pipeline(definition: PipelineDefinition) -> int:
    ctx = _context(definition, _parse_args(definition))
    _write_manifest(ctx, phase="preparing", status="PREPARING")
    try:
        dataset = prepare_data(ctx)
        base_model = prepare_model(ctx)
        method = definition.method_factory(ctx)
        spec = build_spec(ctx, dataset, base_model, method)
        atomic_write_json(ctx.run_dir / "spec.json", spec.model_dump())
        _write_env_manifest(ctx, spec)

        backend = get_backend(spec.backend)
        errors = backend.validate(spec)
        if errors:
            return _finish_validation_failure(ctx, errors)

        job = backend.build(spec)
        atomic_write_json(ctx.run_dir / "job.json", job.model_dump())
        materialize(job)
        _write_env_manifest(ctx, spec, job=job, include_probes=False)
        _write_manifest(ctx, phase="built", status="BUILT", job=job.model_dump())

        if ctx.args.build_only:
            _write_diagnosis(
                ctx,
                status="blocked",
                train_result=None,
                symptom="build-only mode requested; training was not launched",
                root_cause="operator requested --build-only for preflight validation",
                category="environment problem",
                evidence=["spec.json", "job.json", "rendered assets were written"],
                retry_plan="run the same pipeline without --build-only inside a GPU tmux session",
            )
            return 0

        return execute_job(ctx, job)
    except PipelineBlocked as exc:
        _write_manifest(ctx, phase="blocked", status="BLOCKED")
        _write_result(ctx, Status.FAILED, error=str(exc))
        _write_diagnosis(
            ctx,
            status="blocked",
            train_result=None,
            symptom=str(exc),
            root_cause=str(exc),
            category="environment problem",
            evidence=[],
            retry_plan="prepare the missing dependency, then rerun the pipeline",
        )
        return 0 if definition.expected_failure else 1
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        _write_manifest(ctx, phase="crashed", status="FAILED")
        _write_result(ctx, Status.FAILED, error=f"{type(exc).__name__}: {exc}", log_tail=tb)
        _write_diagnosis(
            ctx,
            status="unexpected_failed",
            train_result=None,
            symptom=f"{type(exc).__name__}: {exc}",
            root_cause="pipeline crashed before a TrainResult could be collected",
            category="pipeline code problem",
            evidence=[tb],
            retry_plan="fix the pipeline exception and rerun",
        )
        return 1


def build_spec(
    ctx: PipelineContext, dataset: DatasetRef, base_model: str, method: Any
) -> TrainSpec:
    backend_env = (
        ctx.args.lf_env if ctx.definition.backend == "llamafactory" else ctx.args.verl_env
    )
    # Place the env under the method's env domain (e.g. "llamafactory:ppo"), per capabilities.
    caps = get_backend(ctx.definition.backend).capabilities()
    env_domain = caps.env_domains.get(getattr(method, "kind", ""), ctx.definition.backend)
    extra = {
        "HF_ENDPOINT": ctx.args.hf_endpoint,
        "HF_HOME": str(ctx.cache_root / "huggingface"),
        "HF_HUB_CACHE": str(ctx.cache_root / "huggingface" / "hub"),
        "DATASETS_CACHE": str(ctx.cache_root / "huggingface" / "datasets"),
        "TOKENIZERS_PARALLELISM": "false",
    }
    if not ctx.args.online_train:
        extra["HF_HUB_OFFLINE"] = "1"
        extra["TRANSFORMERS_OFFLINE"] = "1"
    if ctx.args.cuda_visible_devices:
        extra["CUDA_VISIBLE_DEVICES"] = ctx.args.cuda_visible_devices
    bridge_root = ctx.args.bridge_root or os.environ.get("DATAOS_BRIDGE_ROOT")
    if bridge_root:
        extra["DATAOS_BRIDGE_ROOT"] = bridge_root
    extra.update(ctx.definition.env_extra)

    return TrainSpec(
        name=ctx.definition.name,
        backend=ctx.definition.backend,
        base_model=base_model,
        run_dir=str(ctx.run_dir),
        dataset=dataset,
        method=method,
        resources=ctx.definition.resources,
        env=EnvSpec(
            conda_sh=ctx.args.conda_sh,
            backends={env_domain: backend_env},
            deploy_root=None,
            pythonpath=[str(ctx.repo_root / "src"), str(ctx.repo_root)],
            extra=extra,
        ),
    )


def execute_job(ctx: PipelineContext, job: JobSpec) -> int:
    launcher = get_launcher(ctx.args.launcher)
    _write_manifest(ctx, phase="submitting", status="SUBMITTING")
    handle = launcher.submit(job)
    _write_scheduler_state(ctx, status="RUNNING", pid=handle.pid)
    _write_manifest(ctx, phase="running", status="RUNNING", handle=handle.model_dump())

    if ctx.args.submit_only:
        _write_diagnosis(
            ctx,
            status="blocked",
            train_result=None,
            symptom="submit-only mode requested; collection has not run yet",
            root_cause="operator requested --submit-only for asynchronous monitoring",
            category="environment problem",
            evidence=[str(ctx.run_dir / "handle.json")],
            retry_plan=f"poll {ctx.run_dir}/handle.json and then collect",
        )
        return 0

    deadline = time.monotonic() + ctx.args.timeout if ctx.args.timeout else None
    status = launcher.status(handle)
    while not status.is_terminal():
        if deadline and time.monotonic() > deadline:
            launcher.cancel(handle)
            status = Status.TIMEOUT
            break
        time.sleep(ctx.args.poll)
        status = launcher.status(handle)

    result = get_backend(job.backend).collect(job.run_dir)
    result.spec_id = handle.spec_id
    if result.status == Status.COMPLETED and status != Status.COMPLETED:
        result.status = status
    if result.status != Status.COMPLETED and not result.error:
        result.error = f"job ended with status {status.value}"
    if not result.log_tail:
        result.log_tail = launcher.logs(handle, tail=80)
    atomic_write_json(ctx.run_dir / "result.json", result.model_dump())
    _write_scheduler_state(ctx, status=result.status.value, pid=handle.pid)

    ok = _accepted(ctx, result)
    diag_status = _diagnosis_status(ctx, result)
    _write_manifest(
        ctx,
        phase="collected",
        status=result.status.value,
        train_result=result.model_dump(),
        accepted=ok,
    )
    _write_diagnosis_for_result(ctx, result, diag_status)
    return 0 if ok else 1


def prepare_data(ctx: PipelineContext) -> DatasetRef:
    plan = ctx.definition.dataset
    data_dir = ctx.run_dir / "data" / "source"
    out = data_dir / plan.output_name
    sample_size = ctx.args.sample_size or plan.sample_size

    if plan.kind == "local_jsonl":
        src = _resolve_source(ctx, plan.source)
        records = _read_jsonl(src)
        records = _limit(records, sample_size, plan.repeat_to)
        _write_jsonl(out, records)
    elif plan.kind == "hf_preference":
        records = _hf_preference_records(ctx, plan, sample_size)
        _write_jsonl(out, records)
    elif plan.kind == "hf_kto":
        records = _hf_kto_records(ctx, plan, sample_size)
        _write_jsonl(out, records)
    elif plan.kind == "hf_ppo":
        records = _hf_ppo_records(ctx, plan, sample_size)
        _write_jsonl(out, records)
    elif plan.kind == "inline_jsonl":
        records = _limit(list(plan.records), sample_size, plan.repeat_to)
        _write_jsonl(out, records)
    elif plan.kind == "bad_jsonl":
        data_dir.mkdir(parents=True, exist_ok=True)
        lines = plan.raw_lines or ("{bad json",)
        atomic_write_text(out, "\n".join(lines) + "\n")
    else:
        raise ValueError(f"unknown dataset kind: {plan.kind}")

    _write_manifest(
        ctx,
        data={
            "kind": plan.kind,
            "source": plan.source,
            "path": str(out),
            "format": plan.format,
            "sample_size": sample_size,
        },
    )
    return DatasetRef(path=str(out), format=plan.format)


def prepare_model(ctx: PipelineContext) -> str:
    plan = ctx.definition.model
    ref = ctx.args.base_model or os.environ.get(plan.env_var) or plan.default_ref
    maybe_path = Path(ref).expanduser()
    if maybe_path.is_absolute() or ref.startswith("."):
        if maybe_path.exists():
            return str(maybe_path)
        if ctx.args.require_local_model:
            raise PipelineBlocked(f"base model path does not exist: {maybe_path}")
        return ref

    local_dir = ctx.cache_root / "models" / plan.local_subdir
    if (local_dir / "config.json").is_file():
        return str(local_dir)
    if ctx.args.skip_download:
        if ctx.args.require_local_model:
            raise PipelineBlocked(f"model is not cached under {local_dir}")
        return ref

    os.environ["HF_ENDPOINT"] = ctx.args.hf_endpoint
    os.environ["HF_HOME"] = str(ctx.cache_root / "huggingface")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise PipelineBlocked(
            "huggingface_hub is not installed in the preparation environment"
        ) from exc
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=ref, local_dir=str(local_dir))
    return str(local_dir)


def _finish_validation_failure(ctx: PipelineContext, errors: list[str]) -> int:
    error = "validation failed: " + "; ".join(errors)
    (ctx.run_dir / "logs").mkdir(parents=True, exist_ok=True)
    atomic_write_json(ctx.run_dir / "validation_errors.json", {"errors": errors})
    atomic_write_text(ctx.run_dir / "logs" / "validation.out", error + "\n")
    atomic_write_text(
        ctx.run_dir / "run_train.sh",
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "cat logs/validation.out\n"
        "exit 1\n",
    )
    handle = Handle(
        launcher="local",
        spec_id="",
        run_dir=str(ctx.run_dir),
        job_name=ctx.definition.name,
        submitted_at=utc_now_iso(),
        pid=None,
        hostname=socket.gethostname(),
        stdout_path=str(ctx.run_dir / "logs" / "validation.out"),
        exit_code=1,
    )
    atomic_write_json(ctx.run_dir / "handle.json", handle.model_dump())
    result = _write_result(ctx, Status.FAILED, error=error)
    result.artifacts.update(
        {
            "validation_errors": str(ctx.run_dir / "validation_errors.json"),
            "run_train": str(ctx.run_dir / "run_train.sh"),
            "handle": str(ctx.run_dir / "handle.json"),
        }
    )
    atomic_write_json(ctx.run_dir / "result.json", result.model_dump())
    expected = ctx.definition.expected_failure
    _write_scheduler_state(ctx, status="FAILED", pid=None)
    _write_manifest(
        ctx,
        phase="validation_failed",
        status="FAILED",
        train_result=result.model_dump(),
        accepted=expected,
        validation_errors=errors,
    )
    _write_diagnosis(
        ctx,
        status="expected_failed" if expected else "unexpected_failed",
        train_result=result,
        symptom=error,
        root_cause=_classify_validation_root_cause(errors),
        category=_classify_validation_category(errors),
        evidence=errors,
        retry_plan="fix the TrainSpec inputs and rerun",
    )
    return 0 if expected else 1


def _write_result(
    ctx: PipelineContext, status: Status, *, error: str, log_tail: str | None = None
) -> TrainResult:
    result = TrainResult(
        spec_id="",
        status=status,
        base_model=ctx.args.base_model,
        error=error,
        log_tail=log_tail,
        artifacts={"run_dir": str(ctx.run_dir), "logs": str(ctx.run_dir / "logs")},
    )
    atomic_write_json(ctx.run_dir / "result.json", result.model_dump())
    return result


def _accepted(ctx: PipelineContext, result: TrainResult) -> bool:
    if ctx.definition.expected_failure:
        return result.status != Status.COMPLETED
    if result.status != Status.COMPLETED:
        return False
    return all((ctx.run_dir / rel).exists() for rel in ctx.definition.acceptance_artifacts)


def _diagnosis_status(ctx: PipelineContext, result: TrainResult) -> str:
    if ctx.definition.expected_failure and result.status != Status.COMPLETED:
        return "expected_failed"
    if result.status == Status.COMPLETED:
        return "completed"
    return "unexpected_failed"


def _write_diagnosis_for_result(
    ctx: PipelineContext, result: TrainResult, diag_status: str
) -> None:
    text = " ".join(x for x in [result.error or "", result.log_tail or ""] if x)
    root = "training completed and required artifacts are present"
    symptom = "completed"
    retry = "rerun the same pipeline to verify reproducibility"
    evidence: list[str] = []
    if diag_status != "completed":
        category = _classify_failure_text(text, ctx.definition)
        symptom = _failure_symptom(result)
        root = _root_cause_from_category(category, text)
        retry = "apply the fix recommendation, then rerun the same pipeline"
        evidence = [result.error or "", result.log_tail or ""]
    else:
        category = "none"
    _write_diagnosis(
        ctx,
        status=diag_status,
        train_result=result,
        symptom=symptom,
        root_cause=root,
        category=category,
        evidence=evidence,
        retry_plan=retry,
    )


def _failure_symptom(result: TrainResult) -> str:
    text = result.log_tail or ""
    for marker in (
        "JSONDecodeError",
        "KeyError",
        "ValueError",
        "RuntimeError",
        "ImportError",
        "ModuleNotFoundError",
        "CUDA out of memory",
        "No space left on device",
    ):
        if marker not in text:
            continue
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if marker in stripped:
                return stripped
    if result.error and "final_model" not in result.error:
        return result.error
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return result.error or "training did not complete"


def _write_diagnosis(
    ctx: PipelineContext,
    *,
    status: str,
    train_result: TrainResult | None,
    symptom: str,
    root_cause: str,
    category: str,
    evidence: list[str],
    retry_plan: str,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "pipeline": ctx.definition.name,
        "run_dir": str(ctx.run_dir),
        "train_result": train_result.model_dump() if train_result else None,
        "symptom": symptom,
        "root_cause": root_cause,
        "failure_category": category,
        "framework_location": _framework_location(category, symptom),
        "upstream_location": _upstream_location(category, symptom),
        "evidence": [e for e in evidence if e],
        "fix_recommendation": _fix_recommendation(category, symptom),
        "retry_plan": retry_plan,
    }
    issue = _framework_issue_if_needed(ctx, category, symptom)
    if issue:
        payload["framework_issue"] = issue
    atomic_write_json(ctx.run_dir / "diagnosis.json", payload)
    atomic_write_text(ctx.run_dir / "diagnosis.md", _diagnosis_markdown(payload))


def _framework_issue_if_needed(
    ctx: PipelineContext, category: str, symptom: str
) -> dict[str, Any] | None:
    if category != "dataos-train framework problem":
        return None
    return {
        "issue_id": f"dataos-train-{ctx.definition.name}",
        "severity": "P1",
        "affected_pipeline": ctx.definition.name,
        "file": "unknown",
        "function": "unknown",
        "expected_behavior": "pipeline completes or fails with a precise typed diagnostic",
        "actual_behavior": symptom,
        "minimal_repro": f"python pipelines/{ctx.definition.name}.py",
        "proposed_fix": "localize the failing framework path from logs and add a regression test",
        "verification_pipeline": ctx.definition.name,
    }


def _diagnosis_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['pipeline']}",
        "",
        f"- status: {payload['status']}",
        f"- run_dir: {payload['run_dir']}",
        f"- symptom: {payload['symptom']}",
        f"- root_cause: {payload['root_cause']}",
        f"- failure_category: {payload['failure_category']}",
        f"- fix_recommendation: {payload['fix_recommendation']}",
        f"- retry_plan: {payload['retry_plan']}",
    ]
    return "\n".join(lines) + "\n"


def _write_manifest(ctx: PipelineContext, **updates: Any) -> None:
    ctx.manifest.update(updates)
    ctx.manifest["updated_at"] = utc_now_iso()
    atomic_write_json(ctx.run_dir / "manifest.json", ctx.manifest)


def _write_env_manifest(
    ctx: PipelineContext,
    spec: TrainSpec,
    *,
    job: JobSpec | None = None,
    include_probes: bool = True,
) -> None:
    path = ctx.run_dir / "env_manifest.json"
    payload: dict[str, Any]
    if path.is_file():
        try:
            payload = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            payload = {}
    else:
        payload = {}

    payload.update(
        {
            "pipeline": ctx.definition.name,
            "backend": spec.backend,
            "run_dir": str(ctx.run_dir),
            "generated_at": utc_now_iso(),
            "runner": {
                "executable": sys.executable,
                "version": sys.version.split()[0],
                "cwd": str(Path.cwd()),
            },
            "env": {
                "conda_sh": spec.env.conda_sh,
                "backends": dict(spec.env.backends),
                "pythonpath": list(spec.env.pythonpath),
                "extra": _safe_env_extra(spec.env.extra),
            },
        }
    )
    if include_probes:
        payload["runner_probe"] = _probe_python(
            Path(sys.executable),
            _package_specs_for_backend("core"),
            spec,
        )
        payload["backend_probes"] = {
            backend: _probe_python(
                Path(env_path) / "bin" / "python",
                _package_specs_for_backend(backend),
                spec,
            )
            for backend, env_path in spec.env.backends.items()
        }
    if job is not None:
        payload["job"] = {
            "conda_env": job.conda_env,
            "pythonpath": list(job.pythonpath),
            "env": _safe_env_extra(job.env),
            "command": list(job.command),
            "launch_kind": job.launch.kind,
        }
    atomic_write_json(path, payload)
    _write_manifest(ctx, env_manifest=str(path))


def _safe_env_extra(extra: dict[str, str]) -> dict[str, str]:
    safe = {}
    for key, value in extra.items():
        upper = key.upper()
        sensitive = (
            upper.endswith("_TOKEN")
            or "API_KEY" in upper
            or "SECRET" in upper
            or "PASSWORD" in upper
            or "CREDENTIAL" in upper
            or "PROXY" in upper
        )
        safe[key] = "<redacted>" if sensitive else value
    return safe


def _package_specs_for_backend(backend: str) -> list[dict[str, str]]:
    common = [
        {"module": "dataos_train", "dist": "dataos-train"},
        {"module": "torch", "dist": "torch"},
        {"module": "transformers", "dist": "transformers"},
        {"module": "datasets", "dist": "datasets"},
        {"module": "huggingface_hub", "dist": "huggingface-hub"},
        {"module": "numpy", "dist": "numpy"},
    ]
    if backend == "llamafactory":
        return [
            *common,
            {"module": "llamafactory", "dist": "llamafactory"},
            {"module": "trl", "dist": "trl"},
            {"module": "peft", "dist": "peft"},
            {"module": "accelerate", "dist": "accelerate"},
            {"module": "bitsandbytes", "dist": "bitsandbytes"},
        ]
    if backend == "verl":
        return [
            *common,
            {"module": "verl", "dist": "verl"},
            {"module": "vllm", "dist": "vllm"},
            {"module": "ray", "dist": "ray"},
            {"module": "xformers", "dist": "xformers"},
            {"module": "flash_attn", "dist": "flash-attn"},
            {"module": "pyarrow", "dist": "pyarrow"},
        ]
    return common


def _probe_python(
    python: Path,
    packages: list[dict[str, str]],
    spec: TrainSpec,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": str(python),
        "exists": python.is_file(),
    }
    if not python.is_file():
        payload["error"] = "python executable does not exist"
        return payload

    env = os.environ.copy()
    env.update({k: str(v) for k, v in spec.env.extra.items()})
    pythonpath = os.pathsep.join([*spec.env.pythonpath, env.get("PYTHONPATH", "")]).strip(
        os.pathsep
    )
    if pythonpath:
        env["PYTHONPATH"] = pythonpath

    timeout_s = float(os.environ.get("DATAOS_TRAIN_ENV_PROBE_TIMEOUT", "90"))
    try:
        completed = subprocess.run(
            [str(python), "-c", _ENV_PROBE_SCRIPT, json.dumps(packages)],
            check=False,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        payload["error"] = f"environment probe timed out after {timeout_s:.0f}s"
        return payload
    except OSError as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload

    payload["returncode"] = completed.returncode
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("__DATAOS_ENV_PROBE__"):
            try:
                payload.update(json.loads(line.removeprefix("__DATAOS_ENV_PROBE__")))
            except json.JSONDecodeError as exc:
                payload["parse_error"] = f"{type(exc).__name__}: {exc}"
            break
    else:
        payload["error"] = "environment probe did not emit JSON payload"

    stdout_noise = [
        line
        for line in completed.stdout.splitlines()
        if not line.startswith("__DATAOS_ENV_PROBE__")
    ]
    if stdout_noise:
        payload["stdout_tail"] = "\n".join(stdout_noise[-20:])
    if completed.stderr:
        payload["stderr_tail"] = "\n".join(completed.stderr.splitlines()[-20:])
    return payload


_ENV_PROBE_SCRIPT = r"""
from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import platform
import sys

packages = json.loads(sys.argv[1])
out = {
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "platform": platform.platform(),
    "packages": {},
}
for item in packages:
    module_name = item["module"]
    dist_name = item.get("dist") or module_name
    record = {"module": module_name, "distribution": dist_name}
    try:
        module = importlib.import_module(module_name)
        record["importable"] = True
        version = getattr(module, "__version__", None)
        if not version:
            try:
                version = metadata.version(dist_name)
            except metadata.PackageNotFoundError:
                version = None
        record["version"] = version
    except Exception as exc:  # noqa: BLE001
        record["importable"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
    out["packages"][module_name] = record
try:
    import torch

    out["cuda"] = {
        "torch_cuda": getattr(torch.version, "cuda", None),
        "available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
except Exception as exc:  # noqa: BLE001
    out["cuda"] = {"error": f"{type(exc).__name__}: {exc}"}
print("__DATAOS_ENV_PROBE__" + json.dumps(out, sort_keys=True))
"""


def _write_scheduler_state(ctx: PipelineContext, *, status: str, pid: int | None = None) -> None:
    path = ctx.runs_root / "scheduler_state.json"
    state = {"runs": {}}
    if path.is_file():
        try:
            state = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            state = {"runs": {}}
    state.setdefault("runs", {})[ctx.definition.name] = {
        "pipeline": ctx.definition.name,
        "run_dir": str(ctx.run_dir),
        "gpu_ids": _gpu_ids(ctx.args.cuda_visible_devices),
        "pid": pid,
        "status": status,
        "started_at": ctx.manifest.get("started_at"),
        "updated_at": utc_now_iso(),
    }
    atomic_write_json(path, state)


def _context(definition: PipelineDefinition, args: argparse.Namespace) -> PipelineContext:
    pwd = Path(os.environ.get("PWD", "")).expanduser()
    if (pwd / "src" / "dataos_train").is_dir() and (pwd / "pipelines").is_dir():
        repo_root = pwd
    else:
        repo_root = Path(__file__).absolute().parents[1]
    workspace_root = Path(args.work_dir).expanduser() if args.work_dir else repo_root.parent
    runs_root = Path(args.runs_root).expanduser() if args.runs_root else repo_root / "runs"
    run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{definition.name}"
    run_dir = runs_root / run_id
    if args.cache_root:
        cache_root = Path(args.cache_root).expanduser()
    else:
        cache_root = workspace_root / ".cache"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    manifest = {
        "pipeline": definition.name,
        "backend": definition.backend,
        "run_dir": str(run_dir),
        "workspace_root": str(workspace_root),
        "repo_root": str(repo_root),
        "cache_root": str(cache_root),
        "host": socket.gethostname(),
        "gpu_ids": _gpu_ids(args.cuda_visible_devices),
        "expected_failure": definition.expected_failure,
        "expected_failure_reason": definition.expected_failure_reason,
        "started_at": utc_now_iso(),
    }
    return PipelineContext(
        definition=definition,
        args=args,
        repo_root=repo_root,
        workspace_root=workspace_root,
        runs_root=runs_root,
        run_dir=run_dir,
        cache_root=cache_root,
        manifest=manifest,
    )


def _parse_args(definition: PipelineDefinition) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=definition.name)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--reward-model", default=None)
    parser.add_argument(
        "--conda-sh", default=os.environ.get("DATAOS_TRAIN_CONDA_SH", DEFAULT_CONDA_SH)
    )
    parser.add_argument(
        "--lf-env",
        default=os.environ.get("DATAOS_TRAIN_LF_ENV", definition.backend_env or DEFAULT_LF_ENV),
    )
    parser.add_argument(
        "--verl-env", default=os.environ.get("DATAOS_TRAIN_VERL_ENV", DEFAULT_VERL_ENV)
    )
    parser.add_argument("--bridge-root", default=None)
    parser.add_argument("--launcher", choices=["local"], default="local")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--poll", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--require-local-model", action="store_true")
    parser.add_argument("--online-train", action="store_true")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT))
    parser.add_argument("--cache-root", default=None)
    return parser.parse_args()


def _resolve_source(ctx: PipelineContext, source: str) -> Path:
    candidates = []
    p = Path(source).expanduser()
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([ctx.workspace_root / source, ctx.repo_root / source])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise PipelineBlocked(f"dataset source not found: {source}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    atomic_write_text(path, text)


def _limit(
    records: list[dict[str, Any]], sample_size: int, repeat_to: int | None
) -> list[dict[str, Any]]:
    if sample_size:
        records = records[:sample_size]
    if repeat_to and records:
        repeated: list[dict[str, Any]] = []
        while len(repeated) < repeat_to:
            repeated.extend(records)
        records = repeated[:repeat_to]
    return records


def _load_dataset(
    ctx: PipelineContext, plan: DatasetPlan, sample_size: int
) -> list[dict[str, Any]]:
    if ctx.args.skip_download:
        raise PipelineBlocked("HF dataset preparation requested with --skip-download")
    os.environ["HF_ENDPOINT"] = ctx.args.hf_endpoint
    os.environ["HF_HOME"] = str(ctx.cache_root / "huggingface")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise PipelineBlocked("datasets is not installed in the preparation environment") from exc
    ds = load_dataset(
        plan.source,
        split=plan.split,
        cache_dir=str(ctx.cache_root / "huggingface" / "datasets"),
    )
    return [dict(row) for row in ds.select(range(min(sample_size, len(ds))))]


def _hf_preference_records(
    ctx: PipelineContext, plan: DatasetPlan, sample_size: int
) -> list[dict[str, Any]]:
    rows = _load_dataset(ctx, plan, sample_size)
    return [
        {
            "prompt": _extract_prompt(row),
            "chosen": _extract_response(row.get("chosen")),
            "rejected": _extract_response(row.get("rejected")),
        }
        for row in rows
    ]


def _hf_kto_records(
    ctx: PipelineContext, plan: DatasetPlan, sample_size: int
) -> list[dict[str, Any]]:
    rows = _load_dataset(ctx, plan, max(1, (sample_size + 1) // 2))
    out: list[dict[str, Any]] = []
    for row in rows:
        prompt = _extract_prompt(row)
        out.append(
            {"prompt": prompt, "response": _extract_response(row.get("chosen")), "kto_tag": True}
        )
        out.append(
            {"prompt": prompt, "response": _extract_response(row.get("rejected")), "kto_tag": False}
        )
    return out[:sample_size]


def _hf_ppo_records(
    ctx: PipelineContext, plan: DatasetPlan, sample_size: int
) -> list[dict[str, Any]]:
    rows = _load_dataset(ctx, plan, sample_size)
    return [{"prompt": _extract_prompt(row)} for row in rows]


def _extract_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt") or row.get("instruction") or row.get("question")
    if isinstance(prompt, str):
        return prompt
    for key in ("chosen", "rejected", "messages"):
        value = row.get(key)
        if isinstance(value, list):
            for item in value:
                role = item.get("role") or item.get("from") if isinstance(item, dict) else None
                if role in ("user", "human"):
                    content = (
                        item.get("content", item.get("value", ""))
                        if isinstance(item, dict)
                        else ""
                    )
                    if isinstance(content, str):
                        return content
    return ""


def _extract_response(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content", value.get("value", ""))
        return content if isinstance(content, str) else str(content)
    if isinstance(value, list):
        for item in reversed(value):
            if not isinstance(item, dict):
                continue
            role = item.get("role") or item.get("from")
            if role in ("assistant", "gpt"):
                content = item.get("content", item.get("value", ""))
                return content if isinstance(content, str) else str(content)
        if value and isinstance(value[-1], dict):
            content = value[-1].get("content", value[-1].get("value", ""))
            return content if isinstance(content, str) else str(content)
    return ""


def _gpu_ids(raw: str | None) -> list[int]:
    if not raw:
        return []
    ids: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    return ids


def _classify_validation_category(errors: list[str]) -> str:
    joined = " ".join(errors)
    if "reward_model" in joined or "unknown reward.ref" in joined:
        return "pipeline code problem"
    if "dataset is required" in joined:
        return "pipeline code problem"
    return "dataos-train framework problem"


def _classify_validation_root_cause(errors: list[str]) -> str:
    joined = " ".join(errors)
    if "reward_model" in joined:
        return "PPO requires an explicit reward model and the pipeline did not provide one"
    if "unknown reward.ref" in joined:
        return "GRPO reward.ref does not exist in the registered reward set"
    return "TrainSpec failed dataos-train backend validation"


def _classify_failure_text(text: str, definition: PipelineDefinition) -> str:
    low = text.lower()
    if definition.expected_failure:
        if "json" in low or "preference records require" in low or "kto_tag" in low:
            return "data problem"
        if "reward" in low:
            return "pipeline code problem"
    if "converter.py" in low and (
        "keyerror: 'history'" in low
        or 'keyerror: "history"' in low
        or "keyerror: 'system'" in low
        or 'keyerror: "system"' in low
    ):
        return "dataos-train framework problem"
    if (
        "attention heads" in low
        and "tensor parallel size" in low
        or "tensor_model_parallel_size" in low
        and "attention heads" in low
    ):
        return "pipeline code problem"
    if (
        "cuda out of memory" in low
        or "no space left" in low
        or "not installed" in low
        or "modulenotfounderror" in low
        or "no module named" in low
    ):
        return "environment problem"
    if "llamafactory" in low or "verl" in low or "vllm" in low:
        return "LLaMA-Factory / verl upstream problem"
    if "dataos_train" in low:
        return "dataos-train framework problem"
    return "environment problem"


def _root_cause_from_category(category: str, text: str) -> str:
    if category == "data problem":
        return "input data shape is intentionally invalid for this failure-path pipeline"
    if category == "pipeline code problem":
        low = text.lower()
        if "attention heads" in low and "tensor parallel size" in low:
            return (
                "rollout.tensor_parallel_size is incompatible with the selected model's "
                "attention head count"
            )
        return "pipeline configuration is invalid for the requested training method"
    if category == "LLaMA-Factory / verl upstream problem":
        return "training backend rejected the rendered config or failed during execution"
    if category == "dataos-train framework problem":
        low = text.lower()
        if "converter.py" in low and (
            "keyerror: 'history'" in low
            or 'keyerror: "history"' in low
            or "keyerror: 'system'" in low
            or 'keyerror: "system"' in low
        ):
            return (
                "dataos-train declared optional LLaMA-Factory dataset columns that were missing "
                "from converted preference/KTO/PPO rows"
            )
        return "dataos-train produced an invalid render or insufficient diagnostic"
    if "flash_attn" in text:
        return "verl environment is missing flash_attn, which verl imports in the FSDP actor worker"
    return "environment or dependency setup prevented training completion"


def _framework_location(category: str, symptom: str) -> str | None:
    if category == "dataos-train framework problem":
        return "dataos_train backend/build/collect path; inspect logs for exact file"
    if "validation failed" in symptom:
        return "dataos_train.backends.*.validate"
    return None


def _upstream_location(category: str, symptom: str) -> str | None:
    if category == "LLaMA-Factory / verl upstream problem":
        return "llamafactory-cli or verl.trainer.main_ppo"
    return None


def _fix_recommendation(category: str, symptom: str) -> str:
    if category == "none":
        return "no fix required"
    if category == "data problem":
        return "repair the canonical dataset fields and rerun conversion"
    if category == "pipeline code problem":
        return "fix the pipeline TrainSpec inputs before launching GPU work"
    if category == "dataos-train framework problem":
        return "add a framework regression test and patch the failing render/collect path"
    if category == "LLaMA-Factory / verl upstream problem":
        return "minimize the backend-native repro and adjust typed render or dependency versions"
    return "prepare the missing environment dependency and rerun"


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
