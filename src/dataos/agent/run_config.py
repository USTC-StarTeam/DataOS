"""Single-file PTB run configuration.

This module is the user-facing config boundary for the agent-version PTB branch.
The JSON file describes facts needed to start a run; search strategy stays in the
leader plan.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
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


# Training backends the brain knows how to map to envs (mirrors dataos-train's backend
# registry; method→env lookup itself happens in dataos-train's env_for).
_BACKENDS = ("llamafactory", "verl")
DEFAULT_BACKEND = "llamafactory"  # plan 未指明训练方法时的兜底(不是部署事实,是代码默认)


def _tool_path(env: dict[str, Any], name: str) -> str:
    t = (env.get("tools") or {}).get(name, {})
    return t.get("path", "") if isinstance(t, dict) else str(t or "")


def _provides(env: dict[str, Any], name: str) -> list[str]:
    t = (env.get("tools") or {}).get(name, {})
    return list(t.get("provides") or []) if isinstance(t, dict) else []


def env_menu(env: dict[str, Any]) -> dict[str, str]:
    """The full training-env menu {backend: env path}, DERIVED from tools.*.provides (the
    single source) — injected into the TrainSpec's env.backends for env_for() to resolve."""
    return {
        backend: _tool_path(env, name)
        for name in (env.get("tools") or {})
        for backend in _BACKENDS
        if backend in _provides(env, name)
    }


def train_env_path(env: dict[str, Any]) -> str:
    """The default backend's env path (DEFAULT_BACKEND); per-method resolution is
    dataos-train's env_for over the menu, not the brain's business."""
    return env_menu(env).get(DEFAULT_BACKEND, "")


def train_cli_path(env: dict[str, Any]) -> str:
    """Where the dataos-train CLI lives: the thin orchestrator env — the entry NAMED
    ``train_cli_env`` when it provides dataos-train (heavier envs like sft_env may also
    carry the package for in-job imports; the CLI must stay thin); else the first
    provider; else the default backend env."""
    if "dataos-train" in _provides(env, "train_cli_env"):
        return _tool_path(env, "train_cli_env")
    for name in (env.get("tools") or {}):
        if "dataos-train" in _provides(env, name):
            return _tool_path(env, name)
    return train_env_path(env)


def _slurm_facts(data: dict[str, Any]) -> dict[str, Any]:
    """The `slurm:` facts in effect for the partition this run chose.

    Not the raw block: almost everything a scheduler enforces is per-partition, and reading the
    flat block gave `debug` the `gpu` partition's node pool, time limit and allowed QOS. The
    merge rule lives in :func:`dataos_launch.placement.partition_facts`, once, so the brain and
    the launcher cannot disagree about which limits a run is under.
    """
    from dataos_launch.placement import partition_facts

    env = _env(_nonempty_str(data["environment"], "environment"))
    chosen = (data.get("exec") or {}).get("resource") or {}
    return partition_facts(env, chosen.get("partition"))


def _partitions_available(data: dict[str, Any]) -> list[str]:
    from dataos_launch.placement import partitions_available

    return partitions_available(_env(_nonempty_str(data["environment"], "environment")))


def _cluster(env: dict[str, Any], resource: "ResourceConfig") -> dict[str, Any]:
    """Where the job runs, as flat facts. A local environment leaves the ssh/Slurm half None —
    :func:`dataos.agent.runtime._missing_runtime_config_keys` requires them only for kind=slurm."""
    from dataos_launch.placement import partition_facts

    access = env.get("access") or {}
    slurm = partition_facts(env, resource.partition)  # per-partition, not the flat block
    return {
        "host": access.get("host"),
        "user": access.get("user"),
        "port": access.get("port", 22),
        "key": access.get("key_ref"),
        "partition": slurm.get("partition"),  # resolved above: run first, env second
        "qos": resource.qos,   # the run's choice; env.yaml only declares qos_available
        "nodelist": resource.node,
        "gres": f"gpu:{resource.gpus}",
        "cpus_per_task": resource.cpus,
        # Machine facts the prompt layer states as constraints. They live here as NUMBERS rather
        # than as prose in env.yaml's old `discipline` list, because a constraint is knowledge
        # about the machine and belongs where the other prompt text is — reviewed, golden-tested,
        # written once. The list was neither: two environments carried 12 near-identical rules,
        # 1807 characters, differing in one line, and it was re-sent on every experiment turn.
        "mem_per_gpu_mb": slurm.get("mem_per_gpu_mb"),
        "cpus_per_gpu": slurm.get("cpus_per_gpu"),
        "gres_required": slurm.get("gres_required"),
        "node_pool": slurm.get("nodes"),
        "nodes_without_libcuda_link": slurm.get("nodes_without_libcuda_link"),
        "guard_qos": slurm.get("guard_qos"),
    }


def _runtime_paths(env: dict[str, Any], *, task: str, fixed_pool: str,
                   deploy_root: str, runs_base: str) -> dict[str, str]:
    runtime_root = f"{deploy_root}/configs"  # exec-host task assets = deploy configs
    return {
        # run-level facts (from the run JSON): which deploy checkout, where artifacts land
        "deploy_root": deploy_root,
        "runtime_root": runtime_root,  # derived
        "runs_base": runs_base,
        "benchmarks": f"{runtime_root}/tasks/{task}/benchmarks",  # derived
        # the run's input artifact (from the run JSON, like base_model_path)
        "fixed_pool": fixed_pool,
        "train_env": train_env_path(env),
        "eval_env": _tool_path(env, "eval_env"),
        # the env menu, visible to the experimenter for ad-hoc steps (data build / rl / eval)
        "sft_env": _tool_path(env, "sft_env"),
        "rl_env": _tool_path(env, "rl_env"),
        "train_cli_env": train_cli_path(env),
    }


@dataclass(slots=True)
class ResourceConfig:
    gpus: int
    cpus: int
    node: str | None = None  # optional pin; None = let Slurm pick any free node
    # Optional; env.yaml's slurm.partition is the default. Validated against
    # slurm.partitions_available when the environment declares one, because a partition and a
    # QOS are not independent — PTB's `debug` accepts only `debug_normal`, and the pair that
    # does not match is refused by sbatch with a message about the QOS, not about the partition.
    partition: str | None = None
    # Required exactly when the environment declares any, and deliberately not defaulted in
    # env.yaml. The QOS is what the run buys with — it changes between batches, and while it sat
    # in env.yaml with no run-level override the only way to switch was to switch ENVIRONMENTS,
    # which is how two files came to be identical but for that one line. env.yaml says which QOS
    # exist; the run says which one it wants. A workstation declares none, so there is nothing to
    # choose: demanding it there made the LOCAL environment unloadable, which is the same mistake
    # in the other direction — a Slurm word required of a machine that has no Slurm.
    qos: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, qos_available: list[str] | None = None,
                  partitions_available: list[str] | None = None):
        _require(data, ["gpus", "cpus"], "resource")
        partition = _nonempty_str(data["partition"], "resource.partition") \
            if data.get("partition") else None
        if partition and partitions_available and partition not in partitions_available:
            raise ValueError(
                f"resource.partition {partition!r} is not one of the environment's "
                f"partitions_available ({', '.join(partitions_available)})"
            )
        qos = None
        if qos_available:
            _require(data, ["qos"], "resource")
            qos = _nonempty_str(data["qos"], "resource.qos")
            if qos not in qos_available:
                raise ValueError(
                    f"resource.qos {qos!r} is not one of the environment's qos_available "
                    f"({', '.join(qos_available)}) — a typo here PENDs forever with no error"
                )
        elif data.get("qos"):
            raise ValueError(
                f"resource.qos {data['qos']!r} is set, but this environment declares no "
                "slurm.qos_available — there is no QOS here to choose from"
            )
        return cls(
            gpus=_positive_int(data["gpus"], "resource.gpus"),
            cpus=_positive_int(data["cpus"], "resource.cpus"),
            node=_nonempty_str(data["node"], "resource.node") if data.get("node") else None,
            qos=qos,
            partition=partition,
        )

    def to_record(self) -> dict[str, Any]:
        return {"gpus": self.gpus, "cpus": self.cpus, "node": self.node,
                "qos": self.qos, "partition": self.partition}




@dataclass(slots=True)
class InputsConfig:
    """This run's input artifacts."""

    base_model_path: str
    fixed_pool: str


@dataclass(slots=True)
class ExecConfig:
    """Where/how the run executes on the cluster."""

    deploy_root: str  # which DataOS deploy checkout the exec host uses ($AGENTVER)
    runs_base: str    # where this run's artifacts land on the exec host
    resource: ResourceConfig


@dataclass(slots=True)
class BudgetConfig:
    time_hours: float
    eval_samples: int = -1  # eval size per experiment; -1 = FULL benchmark (all samples)


@dataclass(slots=True)
class AgentCliConfig:
    """Which code-agent CLI drives the brain's agents (host-side fact)."""

    provider: str | None = None  # dataos_agent registry name (kimi/claude/qwen/...)
    model: str | None = None     # None = the CLI's own default model
    effort: str | None = None    # thinking level (claude: low/medium/high/xhigh/max)
    idle_timeout_s: float = 1800.0  # stall watchdog: kill+resume after Ns of silence (0 disables)


@dataclass(slots=True)
class PtbRunConfig:
    """The layered run config: identity (goal/workspace) -> pointers (environment/task/
    bench) -> inputs -> exec -> budget -> host (monitor_port). The workspace path IS the
    run's identity (there is no separate run_id)."""

    goal: str
    workspace: str
    environment: str  # REQUIRED pointer into configs/environments/<env>/env.yaml
    task: str         # REQUIRED pointer into configs/tasks/<task>/
    bench: str
    inputs: InputsConfig
    exec: ExecConfig
    budget: BudgetConfig
    agent: AgentCliConfig = field(default_factory=AgentCliConfig)  # host-side agent CLI
    monitor_port: int = 8765  # live-monitor HTTP port (host-side fact; 0 disables the monitor)

    _ALLOWED_KEYS = frozenset({
        "goal", "workspace", "environment", "task", "bench",
        "inputs", "exec", "budget", "agent", "monitor_port",
    })

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PtbRunConfig":
        unknown = sorted(set(data) - cls._ALLOWED_KEYS)
        if unknown:
            raise ValueError(
                f"run config has unknown/forbidden keys: {', '.join(unknown)} — the layered "
                "schema is goal/workspace, environment/task/bench, inputs, exec, budget, "
                "monitor_port (everything else resolves from configs/environments + configs/tasks)"
            )
        _require(data, ["goal", "workspace", "environment", "task", "bench",
                        "inputs", "exec", "budget"], "run config")
        inputs = _object(data["inputs"], "inputs")
        _require(inputs, ["base_model_path", "fixed_pool"], "inputs")
        exec_ = _object(data["exec"], "exec")
        _require(exec_, ["deploy_root", "runs_base", "resource"], "exec")
        budget = _object(data["budget"], "budget")
        _require(budget, ["time_hours"], "budget")
        agent = _object(data["agent"], "agent") if "agent" in data else {}
        unknown_agent = sorted(set(agent) - {"provider", "model", "effort", "idle_timeout_s"})
        if unknown_agent:
            raise ValueError(f"agent has unknown keys: {', '.join(unknown_agent)} "
                             "(allowed: provider, model, effort, idle_timeout_s)")
        if "idle_timeout_s" in agent:
            idle_timeout_s = _nonneg_float(agent["idle_timeout_s"], "agent.idle_timeout_s")
        else:
            idle_timeout_s = 1800.0
        return cls(
            goal=_nonempty_str(data["goal"], "goal"),
            workspace=_nonempty_str(data["workspace"], "workspace"),
            environment=_nonempty_str(data["environment"], "environment"),
            task=_nonempty_str(data["task"], "task"),
            bench=_nonempty_str(data["bench"], "bench"),
            inputs=InputsConfig(
                base_model_path=_nonempty_str(inputs["base_model_path"], "inputs.base_model_path"),
                fixed_pool=_nonempty_str(inputs["fixed_pool"], "inputs.fixed_pool"),
            ),
            exec=ExecConfig(
                deploy_root=_nonempty_str(exec_["deploy_root"], "exec.deploy_root"),
                runs_base=_nonempty_str(exec_["runs_base"], "exec.runs_base"),
                resource=ResourceConfig.from_dict(
                    _object(exec_["resource"], "exec.resource"),
                    qos_available=_slurm_facts(data).get("qos_available"),
                    partitions_available=_partitions_available(data),
                ),
            ),
            budget=BudgetConfig(
                time_hours=_positive_float(budget["time_hours"], "budget.time_hours"),
                eval_samples=_eval_samples(budget["eval_samples"], "budget.eval_samples")
                if "eval_samples" in budget
                else -1,
            ),
            agent=AgentCliConfig(
                provider=_nonempty_str(agent["provider"], "agent.provider")
                if agent.get("provider")
                else None,
                model=_nonempty_str(agent["model"], "agent.model")
                if agent.get("model")
                else None,
                effort=_nonempty_str(agent["effort"], "agent.effort")
                if agent.get("effort")
                else None,
                idle_timeout_s=idle_timeout_s,
            ),
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
    def project_id(self) -> str:
        """The run's identity = the workspace basename (suffix included, e.g. 'ptb-x(1)')."""
        return self.workspace_path.name

    @property
    def workspace_root(self) -> str:
        return str(self.workspace_path.parent)

    def to_record(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "workspace": self.workspace,
            "environment": self.environment,
            "task": self.task,
            "bench": self.bench,
            "inputs": {"base_model_path": self.inputs.base_model_path,
                       "fixed_pool": self.inputs.fixed_pool},
            "exec": {"deploy_root": self.exec.deploy_root,
                     "runs_base": self.exec.runs_base,
                     "resource": self.exec.resource.to_record()},
            "budget": {"time_hours": self.budget.time_hours,
                       "eval_samples": self.budget.eval_samples},
            "agent": {k: v for k, v in
                      {"provider": self.agent.provider, "model": self.agent.model,
                       "effort": self.agent.effort,
                       "idle_timeout_s": self.agent.idle_timeout_s}.items()
                      if v is not None},
            "monitor_port": self.monitor_port,
        }

    def runtime_config(self) -> dict[str, Any]:
        env = _env(self.environment)
        return {
            "environment": self.environment,
            # The environment's own `kind` travels with the config, because what a runtime must
            # provide differs by it: ssh coordinates and a Slurm partition are facts about
            # reaching another machine, and a workstation environment has neither.
            "kind": str(env.get("kind") or "slurm"),
            "task": self.task,
            "runtime_ref": DEFAULT_RUNTIME_REF,
            "cluster": _cluster(env, self.exec.resource),
            "paths": _runtime_paths(
                env, task=self.task, fixed_pool=self.inputs.fixed_pool,
                deploy_root=self.exec.deploy_root, runs_base=self.exec.runs_base,
            ),
            "gateway": dict(env.get("gateway") or {}),  # connection facts only
        }

    def resolved_record(self, *, source_config: Path) -> dict[str, Any]:
        return {
            "run": {
                "project_id": self.project_id,
                "workspace": str(self.workspace_path),
                "source_config": str(source_config),
                "goal": self.goal,
                "time_hours": self.budget.time_hours,
            },
            "posttrainbench": {"bench": self.bench, "profile": "formal"},
            "model": {"base_model_path": self.inputs.base_model_path},
            "resource": self.exec.resource.to_record(),
            "runtime": self.runtime_config(),
        }

    def to_auto_config(self, *, resolved_config_path: Path) -> AutoRunConfig:
        return AutoRunConfig(
            project_id=self.project_id,
            instruct=self.goal,
            goal=self.goal,
            scenario="posttrainbench",
            criterion=Criterion(metric="score", threshold=None, direction="max"),
            budget=Budget(max_wall_clock_s=self.budget.time_hours * 3600.0),
            workspace=self.workspace_root,
            backend=os.environ.get("DATAOS_BACKEND", "cli"),
            # precedence: run JSON `agent` block > env var > default (kimi / CLI default model)
            agent_provider=(
                self.agent.provider
                or os.environ.get("DATAOS_AGENT_PROVIDER")
                or "kimi"
            ),
            agent_model=(
                self.agent.model
                or os.environ.get("DATAOS_AGENT_MODEL")
                or None  # None = the CLI's own default model
            ),
            agent_effort=self.agent.effort,
            agent_idle_timeout_s=self.agent.idle_timeout_s,
            runtime_config=self.runtime_config(),
            runtime_config_path=str(resolved_config_path),
            h100_base_model=self.inputs.base_model_path,
            base_model_id=_model_id_from_path(self.inputs.base_model_path),
            target_bench=self.bench,
            ptb_profile="formal",
            eval_limit=self.budget.eval_samples,
            monitor_port=self.monitor_port,
        )


def _fresh_workspace(requested: Path) -> Path:
    """First non-existing variant of the requested workspace dir: itself, then (1), (2), …"""
    if not requested.exists():
        return requested
    for i in range(1, 100):
        cand = requested.with_name(f"{requested.name}({i})")
        if not cand.exists():
            return cand
    raise ValueError(f"no free workspace name after {requested.name}(99)")


def load_auto_run_config(config_path: str | Path, *, resume: bool = False) -> AutoRunConfig:
    source = Path(config_path).expanduser().resolve()
    spec = PtbRunConfig.from_json(source)
    if resume:
        run_dir = spec.workspace_path
        if not run_dir.is_dir():
            raise ValueError(f"--resume given but workspace does not exist: {run_dir}")
    else:
        run_dir = _fresh_workspace(spec.workspace_path)
        if run_dir != spec.workspace_path:
            spec.workspace = str(run_dir)
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
    """eval size per experiment: positive int (subset), -1 (full), or -2 (auto: the
    experimenter picks the size per experiment, default full)."""
    if not isinstance(value, int) or isinstance(value, bool) or (value <= 0 and value not in (-1, -2)):
        raise ValueError(
            f"{field_name} must be a positive integer, -1 (full benchmark), or -2 (auto)"
        )
    return value


def _nonneg_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _nonneg_float(value: Any, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative number")
    return float(value)


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
