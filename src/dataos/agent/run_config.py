"""Single-file PTB run configuration.

This module is the user-facing config boundary for the agent-version PTB branch.
The JSON file describes facts needed to start a run; search strategy stays in the
leader plan.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dataos.core.io import atomic_write_json

from .config import AutoRunConfig, Budget, Criterion

# ── Unified config (configs/, in-repo) ─────────────────────────────────────────
# ALL deployment/task facts live in this repo: configs/environments/<env>/env.yaml
# (machine, tools, train backend binding, ops), configs/tasks/<task>/ (task contract +
# eval bridge + benchmark data). A run JSON carries only run-level facts + the
# environment/task pointers — every other fact resolves through them. There is no
# external runtime checkout anymore. Field spec: configs/CONTRACT.md.
_REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = _REPO_ROOT / "configs"

DEFAULT_ENVIRONMENT = "PTB_H100_slurm"
DEFAULT_TASK = "posttrainbench"
DEFAULT_RUNTIME_REF = "main"  # compose run-config field; kept for template compatibility


@lru_cache(maxsize=None)
def _env(environment: str) -> dict[str, Any]:
    """Deployment facts from configs/environments/<env>/env.yaml (the only source)."""
    path = CONFIGS_DIR / "environments" / environment / "env.yaml"
    if not path.is_file():
        raise ValueError(f"unknown environment {environment!r}: no {path}")
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return data


def _tool_path(env: dict[str, Any], name: str) -> str:
    t = (env.get("tools") or {}).get(name, {})
    return t.get("path", "") if isinstance(t, dict) else str(t or "")


def train_env_path(env: dict[str, Any]) -> str:
    """SFT runtime path from the env's train block: train.backends[train.backend] ->
    tools.<key>.path (the backend↔env binding is declared exactly once, in env.yaml)."""
    train = env.get("train") or {}
    backend = train.get("backend", "llamafactory")
    tool_key = (train.get("backends") or {}).get(backend)
    if not tool_key:
        raise ValueError(f"env.yaml train.backends has no entry for backend {backend!r}")
    return _tool_path(env, tool_key)


def _cluster(env: dict[str, Any], resource: "ResourceConfig") -> dict[str, Any]:
    access = env.get("access") or {}
    slurm = env.get("slurm") or {}
    return {
        "host": access.get("host"),
        "user": access.get("user"),
        "port": access.get("port", 22),
        "key": access.get("key_ref"),
        "partition": slurm.get("partition"),
        "qos": slurm.get("qos"),
        "nodelist": resource.node,
        "gres": f"gpu:{resource.gpus}",
        "cpus_per_task": resource.cpus,
    }


def _runtime_paths(env: dict[str, Any]) -> dict[str, str]:
    p = env.get("paths") or {}
    deploy_root = p.get("deploy_root", "")
    return {
        "deploy_root": deploy_root,
        "runtime_root": f"{deploy_root}/configs",  # exec-host task assets = deploy configs
        "runs_base": p.get("runs_base", ""),
        "benchmarks": p.get("benchmarks", ""),
        "fixed_pool": p.get("fixed_pool", ""),
        "train_env": train_env_path(env),
        "eval_env": _tool_path(env, "eval_env"),
    }


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




@dataclass(slots=True)
class PtbRunConfig:
    run_id: str
    workspace: str
    goal: str
    bench: str
    base_model_path: str
    time_budget_hours: float
    resource: ResourceConfig
    environment: str  # REQUIRED pointer into configs/environments/<env>/env.yaml
    task: str         # REQUIRED pointer into configs/tasks/<task>/
    eval_samples: int = -1  # eval size per experiment; -1 = FULL benchmark (all samples)
    monitor_port: int = 8765  # live-monitor HTTP port (host-side fact; 0 disables the monitor)

    _ALLOWED_KEYS = frozenset({
        "run_id", "workspace", "goal", "bench", "base_model_path", "time_budget_hours",
        "resource", "environment", "task", "eval_samples", "monitor_port",
    })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PtbRunConfig":
        unknown = sorted(set(data) - cls._ALLOWED_KEYS)
        if unknown:
            raise ValueError(
                f"run config has unknown/forbidden keys: {', '.join(unknown)} — a run JSON "
                "carries only run-level facts + environment/task pointers (paths/train/env "
                "facts resolve from configs/environments + configs/tasks)"
            )
        required = [
            "run_id",
            "workspace",
            "goal",
            "bench",
            "base_model_path",
            "time_budget_hours",
            "resource",
            "environment",
            "task",
        ]
        _require(data, required, "run config")
        return cls(
            run_id=_nonempty_str(data["run_id"], "run_id"),
            workspace=_nonempty_str(data["workspace"], "workspace"),
            goal=_nonempty_str(data["goal"], "goal"),
            bench=_nonempty_str(data["bench"], "bench"),
            base_model_path=_nonempty_str(data["base_model_path"], "base_model_path"),
            time_budget_hours=_positive_float(data["time_budget_hours"], "time_budget_hours"),
            resource=ResourceConfig.from_dict(_object(data["resource"], "resource")),
            environment=_nonempty_str(data["environment"], "environment"),
            task=_nonempty_str(data["task"], "task"),
            eval_samples=_eval_samples(data["eval_samples"], "eval_samples")
            if "eval_samples" in data
            else -1,
            monitor_port=_nonneg_int(data["monitor_port"], "monitor_port")
            if "monitor_port" in data
            else 8765,
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
            "environment": self.environment,
            "task": self.task,
        }

    def runtime_config(self) -> dict[str, Any]:
        env = _env(self.environment)
        gateway = dict(env.get("gateway") or {})
        return {
            "environment": self.environment,
            "task": self.task,
            "runtime_ref": DEFAULT_RUNTIME_REF,
            "cluster": _cluster(env, self.resource),
            "paths": _runtime_paths(env),
            "gateway": gateway,
            "judge": {**gateway, "workers": 32},
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
            agent_provider=os.environ.get("DATAOS_AGENT_PROVIDER", "kimi"),
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
