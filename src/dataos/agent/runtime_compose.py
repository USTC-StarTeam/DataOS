"""Runtime composer — resolve the three-layer config into the two prompt sections.

The unified config lives IN this repo under ``configs/``:

    configs/environments/<env>/env.yaml   deployment facts (machine, tools, train, ops)
    configs/tasks/<task>/task.yaml        task contract (scoring, assets, eval bridge)
    configs/runs/*.json                   per-run facts + environment/task pointers

``resolve()`` fills ``{{env.*}}/{{run.*}}/{{task.*}}`` in the env/task contracts
(``{ctx.*}`` placeholders are preserved for the experimenter to fill at execution
time); ``render()`` turns the resolved contract into the ENV OPS + TASK prompt
sections. Absorbed from DataOS-Runtime's compose.py — no external checkout, no
sys.path games. Only pyyaml + jinja2. Field spec: ``configs/CONTRACT.md``; design:
``docs/runtime-contract.md``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

_REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = _REPO_ROOT / "configs"
_TEMPLATES_DIR = Path(__file__).resolve().parent / "compose_templates"

_JINJA = Environment(undefined=StrictUndefined)   # env/run/task missing field -> error


class _Ctx:
    """Execution-time placeholder proxy: {{ ctx.foo }} renders as the literal {ctx.foo}."""

    def __getattr__(self, name: str) -> str:
        return "{ctx.%s}" % name


def _render_str(s: str, ns: dict) -> str:
    return _JINJA.from_string(s).render(**ns)


def _render_tree(obj, ns):
    if isinstance(obj, str):
        return _render_str(obj, ns)
    if isinstance(obj, dict):
        return {k: _render_tree(v, ns) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_render_tree(v, ns) for v in obj]
    return obj


def _require(d: dict, keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in d or d[k] in (None, "")]
    if missing:
        raise ValueError(f"{where} missing fields: {', '.join(missing)}")


def _validate(env: dict, task: dict) -> None:
    # no `discipline`: constraints are prompt text now (src/dataos/prompt/*/context/
    # cluster_discipline), stated once and golden-tested, not prose duplicated per env.
    _require(env, ["name", "kind", "access", "tools", "operations"], "env.yaml")
    # `submit`/`poll`/`cancel` used to be required here and rendered into the prompt as a
    # ready-to-run ssh+sbatch line. That was a second submission path, advertised by us, next to
    # the three instruments — and an experimenter used it: job 29154 went out through it and
    # hand-re-derived the TMPDIR, offline-HF and venv setup dataos-data already owns. The tools
    # provide submit/status/cancel now, including `dataos-data probe` for a script that produces
    # nothing, so the escape hatch has nothing left to be an escape from.
    for verb in ("fetch", "activate"):
        if verb not in env["operations"]:
            raise ValueError(f"env.yaml.operations missing verb: {verb}")
    _require(task, ["name", "inputs", "objective", "constraints", "produce", "scoring"], "task.yaml")
    sc = task["scoring"]
    _require(sc, ["evaluator", "command", "result"], "task.yaml.scoring")
    _require(sc["result"], ["file", "field", "range"], "task.yaml.scoring.result")


def resolve(run_config: dict, configs_root: str | Path = CONFIGS_DIR) -> dict:
    """Read env.yaml + task.yaml, fill {{env.*}}/{{run.*}}/{{task.*}} (ctx preserved),
    validate, and return the resolved contract."""
    configs_root = Path(configs_root)
    envn = run_config["environment"]
    taskn = run_config["task"]
    task_dir = configs_root / "tasks" / taskn
    env = yaml.safe_load(
        (configs_root / "environments" / envn / "env.yaml").read_text(encoding="utf-8")
    )
    task = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    task["assets_root"] = str(task_dir)
    _validate(env, task)

    resolved_env = dict(env)
    # `slurm:` collapses to the partition this run chose BEFORE anything renders against it, so
    # `ns` below carries the resolved env, not the raw one. Templates and prompts then read one
    # flat block that is true for THIS run, instead of each reader deciding for itself whether a
    # key was machine-wide or partition-scoped — which is how `debug` came to be handed the `gpu`
    # partition's node pool, its 3-day limit and its QOS list.
    if env.get("slurm"):
        from dataos_launch.placement import partition_facts

        resolved_env["slurm"] = partition_facts(
            env, (run_config.get("resource") or {}).get("partition")
        )

    ns = {"env": resolved_env, "run": run_config, "task": task, "ctx": _Ctx()}
    resolved_env["operations"] = {
        k: _render_str(v, ns).strip() for k, v in env["operations"].items()
    }

    resolved_task = dict(task)
    resolved_task["constraints"] = [_render_str(c, ns) for c in task["constraints"]]
    sc = dict(task["scoring"])
    for k in ("command", "judge_command"):
        if sc.get(k):
            sc[k] = _render_str(sc[k], ns).strip()
    sc["result"] = _render_tree(sc["result"], ns)
    resolved_task["scoring"] = sc

    # Per-task layout: a `train/` dir means training is FROZEN (the task fixes the recipe);
    # its absence means training is the agent's method (e.g. dataos-train).
    train_dir = task_dir / "train"
    if train_dir.is_dir():
        train_spec = {"frozen": True}
        train_yaml = train_dir / "train.yaml"
        if train_yaml.is_file():
            spec = yaml.safe_load(train_yaml.read_text(encoding="utf-8")) or {}
            if spec.get("command"):
                spec["command"] = _render_str(spec["command"], ns).strip()
            train_spec.update(spec)
        resolved_task["train"] = train_spec
    else:
        resolved_task["train"] = None  # agent-owned training

    return {
        "runtime_ref": run_config.get("runtime_ref"),
        "environment": resolved_env,
        "task": resolved_task,
        "run": run_config,
    }


def render(resolved: dict) -> tuple[str, str]:
    """Render the resolved contract into the two prompt sections (env ops, task def)."""
    jenv = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)), trim_blocks=True, lstrip_blocks=True
    )
    env_ops = jenv.get_template("environment.j2").render(
        env=resolved["environment"], run=resolved["run"]
    )
    task_def = jenv.get_template("task.j2").render(task=resolved["task"], run=resolved["run"])
    return env_ops, task_def


# --- brain-side wiring ---------------------------------------------------------


def build_run_config(cfg, *, parent_model: str, eval_limit: int) -> dict:
    """Assemble the compose run-config dict from our AutoRunConfig (runtime_config facts)."""
    rt = getattr(cfg, "runtime_config", {}) or {}
    cluster = rt.get("cluster", {})
    paths = rt.get("paths", {})
    gres = str(cluster.get("gres", "gpu:1"))
    try:
        gpus = int(gres.rsplit(":", 1)[-1])
    except ValueError:
        gpus = 1
    return {
        "environment": rt.get("environment", "example-slurm"),
        "task": rt.get("task", "posttrainbench"),
        "runtime_ref": rt.get("runtime_ref", "main"),
        "base_model": parent_model,
        "bench": cfg.target_bench,
        "runs_base": paths.get("runs_base"),
        "deploy_root": paths.get("deploy_root"),
        "resource": {
            "node": cluster.get("nodelist"),
            "gpus": gpus,
            "cpus": cluster.get("cpus_per_task"),
            # every submitter reads the QOS from here now — env.yaml only declares which exist
            "qos": cluster.get("qos"),
            "partition": cluster.get("partition"),
        },
        "eval_limit": eval_limit,
        "fixed_pool": paths.get("fixed_pool"),
        "benchmarks": paths.get("benchmarks"),
    }


def render_sections(cfg, *, parent_model: str, eval_limit: int) -> tuple[str, str, dict]:
    """Return (env_ops, task_def, resolved) — the two prompt sections rendered from the
    resolved contract plus the contract itself (stages/tools/train_spec read it)."""
    rc = build_run_config(cfg, parent_model=parent_model, eval_limit=eval_limit)
    resolved = resolve(rc)
    env_ops, task_def = render(resolved)
    return env_ops, task_def, resolved
