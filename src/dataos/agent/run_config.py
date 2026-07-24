"""Single-file PTB run configuration.

This module is the user-facing config boundary for the agent-version PTB branch.
The JSON file describes facts needed to start a run; search strategy stays in the
main-agent plan.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataos.core.io import atomic_write_json

from .config import AutoRunConfig, Budget, Criterion

# ── Environment archive (DataOS-Runtime) ──────────────────────────────────────
# MACHINE facts (access / slurm partition+qos / runs_base / runtime_root / gateway
# base_url+key_file) live in the runtime repo's environments/<env>/env.yaml; the brain
# reads a LOCAL checkout (DATAOS_RUNTIME_DIR, or a sibling ../DataOS-Runtime).
# NOT in env.yaml (by contract, see DataOS-Runtime/CONTRACT.md): task-data paths
# (fixed_pool from the run JSON, benchmarks defaulted below), deploy_root (DataOS's own
# code root on the exec host — a brain-side fact), and the judge model/workers (a task
# choice — task.yaml scoring.judge; the DEFAULT_JUDGE below only feeds the legacy
# RUNTIME CONFIG block). Hardcoded values are FALLBACKS when no checkout is present.
DEFAULT_ENVIRONMENT = os.environ.get("DATAOS_ENVIRONMENT", "PTB_H100_slurm")
DEFAULT_TASK = os.environ.get("DATAOS_TASK", "posttrainbench")
DEFAULT_RUNTIME_REF = os.environ.get("DATAOS_RUNTIME_REF", "main")


def runtime_dir() -> Path | None:
    """Local checkout of DataOS-Runtime the brain reads env.yaml from, or None."""
    candidates: list[Path] = []
    raw = os.environ.get("DATAOS_RUNTIME_DIR")
    if raw:
        candidates.append(Path(raw).expanduser())
    # sibling of the brain repo: <brain_repo>/../DataOS-Runtime
    candidates.append(Path(__file__).resolve().parents[3].parent / "DataOS-Runtime")
    for cand in candidates:
        if cand and (cand / "environments").is_dir():
            return cand
    return None


def _load_env_yaml(environment: str = DEFAULT_ENVIRONMENT) -> dict[str, Any]:
    root = runtime_dir()
    if root is None:
        return {}
    path = root / "environments" / environment / "env.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


_ENV = _load_env_yaml()
_ENV_ACCESS = _ENV.get("access", {})
_ENV_SLURM = _ENV.get("slurm", {})
_ENV_PATHS = _ENV.get("paths", {})
_ENV_GW = _ENV.get("gateway", {})

DEFAULT_CLUSTER = {
    "host": _ENV_ACCESS.get("host", "114.214.215.159"),
    "user": _ENV_ACCESS.get("user", "hcgu"),
    "port": _ENV_ACCESS.get("port", 22),
    "key": _ENV_ACCESS.get("key_ref", "~/.ssh/ghc_miyao"),
    "partition": _ENV_SLURM.get("partition", "gpu"),
    "qos": _ENV_SLURM.get("qos", "gpu_priv"),
}
DEFAULT_DEPLOY_ROOT = "/data/hcgu/users/tjh/dataos_agentver"  # brain-side fact (DataOS code root on exec host); no longer in env.yaml
DEFAULT_RUNTIME_ROOT = _ENV_PATHS.get("runtime_root", DEFAULT_DEPLOY_ROOT)
DEFAULT_RUNS_BASE = _ENV_PATHS.get("runs_base", "/data/hcgu/users/tjh/dataos_agentver_runs")
DEFAULT_BENCHMARKS = _ENV_PATHS.get(
    "benchmarks", f"{DEFAULT_DEPLOY_ROOT}/resources/data/posttrainbench_benchmarks"
)
DEFAULT_GATEWAY = {
    "base_url": _ENV_GW.get("base_url", "https://api.chatanywhere.tech/v1"),
    "model": "qwen3.5-plus",  # judge/llm-op model: a task choice now (task.yaml scoring.judge); kept for the legacy RUNTIME CONFIG block
    "key_file": _ENV_GW.get("key_file", "/data/hcgu/users/tjh/.chatanywhere_key"),
}
DEFAULT_JUDGE = {
    **DEFAULT_GATEWAY,
    # judge doc concurrency; passed to run_lmeval_judge.py --workers
    "workers": 32,  # judge doc concurrency; task.yaml scoring.judge.workers is authoritative for the composed scoring command
}
# env.yaml is now the source of truth for the data pool + runtimes; run-config `paths` defaults
# from here so a run-config need not repeat them (env is a (task,machine) deployment).
DEFAULT_FIXED_POOL = _ENV_PATHS.get("fixed_pool", "")
_ENV_TOOLS = _ENV.get("tools", {})


def _tool_path(name: str) -> str:
    t = _ENV_TOOLS.get(name, {})
    return t.get("path", "") if isinstance(t, dict) else str(t or "")


DEFAULT_SFT_ENV = _tool_path("sft_env")    # the SFT runtime (uv venv) — fills the legacy train_env slot
DEFAULT_EVAL_ENV = _tool_path("eval_env")  # the eval runtime (uv venv)


@dataclass(slots=True)
class ResourceConfig:
    node: str
    gpus: int
    cpus: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceConfig":
        _require(data, ["node", "gpus", "cpus"], "resource")
        return cls(
            node=_nonempty_str(data["node"], "resource.node"),
            gpus=_positive_int(data["gpus"], "resource.gpus"),
            cpus=_positive_int(data["cpus"], "resource.cpus"),
        )

    def to_record(self) -> dict[str, Any]:
        return {"node": self.node, "gpus": self.gpus, "cpus": self.cpus}

    def to_cluster(self) -> dict[str, Any]:
        return {
            **DEFAULT_CLUSTER,
            "nodelist": self.node,
            "gres": f"gpu:{self.gpus}",
            "cpus_per_task": self.cpus,
        }


@dataclass(slots=True)
class PathConfig:
    """Per-run paths — all DEFAULTED from env.yaml (env is the source of truth for the data pool
    + runtimes). A run-config need not set `paths` at all; override a field only to point THIS run
    at a different pool/runtime than the environment's default. (conda_sh dropped: envs are uv.)"""

    fixed_pool: str = ""
    train_env: str = ""  # SFT runtime = env.yaml tools.sft_env; keeps the name the RUNTIME CONFIG block uses
    eval_env: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PathConfig":
        return cls(
            fixed_pool=_nonempty_str(data["fixed_pool"], "paths.fixed_pool") if data.get("fixed_pool") else DEFAULT_FIXED_POOL,
            train_env=_nonempty_str(data["train_env"], "paths.train_env") if data.get("train_env") else DEFAULT_SFT_ENV,
            eval_env=_nonempty_str(data["eval_env"], "paths.eval_env") if data.get("eval_env") else DEFAULT_EVAL_ENV,
        )

    @classmethod
    def defaults(cls) -> "PathConfig":
        return cls(fixed_pool=DEFAULT_FIXED_POOL, train_env=DEFAULT_SFT_ENV, eval_env=DEFAULT_EVAL_ENV)

    def to_record(self) -> dict[str, Any]:
        return {"fixed_pool": self.fixed_pool, "train_env": self.train_env, "eval_env": self.eval_env}

    def to_runtime_paths(self) -> dict[str, Any]:
        return {
            "deploy_root": DEFAULT_DEPLOY_ROOT,
            "runtime_root": DEFAULT_RUNTIME_ROOT,
            "runs_base": DEFAULT_RUNS_BASE,
            "benchmarks": DEFAULT_BENCHMARKS,
            **self.to_record(),
        }


@dataclass(slots=True)
class PtbRunConfig:
    run_id: str
    workspace: str
    goal: str
    bench: str
    base_model_path: str
    time_budget_hours: float
    resource: ResourceConfig
    paths: PathConfig
    eval_samples: int = -1  # eval size per experiment; -1 = FULL benchmark (all samples)
    monitor_port: int = 8765  # live-monitor HTTP port (host-side fact; 0 disables the monitor)
    environment: str = DEFAULT_ENVIRONMENT  # which DataOS-Runtime environment (env.yaml contract)
    task: str = DEFAULT_TASK  # which DataOS-Runtime task (task.yaml contract + eval bridge)
    runtime_ref: str = DEFAULT_RUNTIME_REF  # DataOS-Runtime tag/commit the exec host checks out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PtbRunConfig":
        required = [
            "run_id",
            "workspace",
            "goal",
            "bench",
            "base_model_path",
            "time_budget_hours",
            "resource",
        ]  # `paths` is optional — defaults from env.yaml (data pool + runtimes)
        _require(data, required, "run config")
        return cls(
            run_id=_nonempty_str(data["run_id"], "run_id"),
            workspace=_nonempty_str(data["workspace"], "workspace"),
            goal=_nonempty_str(data["goal"], "goal"),
            bench=_nonempty_str(data["bench"], "bench"),
            base_model_path=_nonempty_str(data["base_model_path"], "base_model_path"),
            time_budget_hours=_positive_float(data["time_budget_hours"], "time_budget_hours"),
            resource=ResourceConfig.from_dict(_object(data["resource"], "resource")),
            paths=PathConfig.from_dict(_object(data["paths"], "paths"))
            if "paths" in data
            else PathConfig.defaults(),
            eval_samples=_eval_samples(data["eval_samples"], "eval_samples")
            if "eval_samples" in data
            else -1,
            monitor_port=_nonneg_int(data["monitor_port"], "monitor_port")
            if "monitor_port" in data
            else 8765,
            environment=_nonempty_str(data["environment"], "environment")
            if "environment" in data
            else DEFAULT_ENVIRONMENT,
            task=_nonempty_str(data["task"], "task") if "task" in data else DEFAULT_TASK,
            runtime_ref=_nonempty_str(data["runtime_ref"], "runtime_ref")
            if "runtime_ref" in data
            else DEFAULT_RUNTIME_REF,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PtbRunConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("run config JSON must be an object")
        return cls.from_dict(data)

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace).expanduser()

    @property
    def workspace_root(self) -> str:
        path = self.workspace_path
        if path.name != self.run_id:
            raise ValueError("workspace must point to the run directory and end with run_id")
        return str(path.parent if str(path.parent) else Path("."))

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workspace": self.workspace,
            "goal": self.goal,
            "bench": self.bench,
            "base_model_path": self.base_model_path,
            "time_budget_hours": self.time_budget_hours,
            "resource": self.resource.to_record(),
            "paths": self.paths.to_record(),
        }

    def runtime_config(self) -> dict[str, Any]:
        return {
            "environment": self.environment,
            "task": self.task,
            "runtime_ref": self.runtime_ref,
            "cluster": self.resource.to_cluster(),
            "paths": self.paths.to_runtime_paths(),
            "gateway": dict(DEFAULT_GATEWAY),
            "judge": dict(DEFAULT_JUDGE),
        }

    def resolved_record(self, *, source_config: Path) -> dict[str, Any]:
        runtime = self.runtime_config()
        return {
            "run": {
                "run_id": self.run_id,
                "workspace": str(self.workspace_path),
                "source_config": str(source_config),
                "goal": self.goal,
                "time_budget_hours": self.time_budget_hours,
            },
            "posttrainbench": {"bench": self.bench, "profile": "formal"},
            "model": {"base_model_path": self.base_model_path},
            "resource": self.resource.to_record(),
            "runtime": runtime,
        }

    def to_auto_config(self, *, resolved_config_path: Path) -> AutoRunConfig:
        return AutoRunConfig(
            project_id=self.run_id,
            instruct=self.goal,
            goal=self.goal,
            scenario="posttrainbench",
            criterion=Criterion(metric="score", threshold=None, direction="max"),
            budget=Budget(max_wall_clock_s=self.time_budget_hours * 3600.0),
            workspace=self.workspace_root,
            backend=os.environ.get("DATAOS_BACKEND", "cli"),
            agent_provider=os.environ.get("DATAOS_AGENT_PROVIDER", "claude"),
            agent_model=os.environ.get("DATAOS_AGENT_MODEL") or None,  # None = CLI default
            runtime_config=self.runtime_config(),
            runtime_config_path=str(resolved_config_path),
            h100_base_model=self.base_model_path,
            base_model_id=_model_id_from_path(self.base_model_path),
            target_bench=self.bench,
            ptb_profile="formal",
            eval_limit=self.eval_samples,
            monitor_port=self.monitor_port,
        )


def load_auto_run_config(config_path: str | Path) -> AutoRunConfig:
    source = Path(config_path).expanduser().resolve()
    spec = PtbRunConfig.from_json(source)
    run_dir = spec.workspace_path
    run_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = run_dir / "run_config.json"
    resolved_path = run_dir / "resolved_config.json"
    atomic_write_json(normalized_path, spec.to_record())
    atomic_write_json(resolved_path, spec.resolved_record(source_config=source))
    return spec.to_auto_config(resolved_config_path=resolved_path)


def _require(data: dict[str, Any], fields: list[str], context: str) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f"{context} missing required fields: {', '.join(missing)}")


def _object(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def _nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _eval_samples(value: Any, field_name: str) -> int:
    """eval size per experiment: a positive int (subset) or -1 (full benchmark)."""
    if not isinstance(value, int) or isinstance(value, bool) or (value <= 0 and value != -1):
        raise ValueError(f"{field_name} must be a positive integer or -1 (full benchmark)")
    return value


def _nonneg_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return float(value)


def _model_id_from_path(path: str) -> str | None:
    parts = Path(path).parts
    for index, part in enumerate(parts):
        if part.startswith("models--"):
            name = part.removeprefix("models--").replace("--", "/")
            return name or None
        if part == "hub" and index + 1 < len(parts) and parts[index + 1].startswith("models--"):
            return parts[index + 1].removeprefix("models--").replace("--", "/")
    return Path(path).name or None
