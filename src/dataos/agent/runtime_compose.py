"""C4 wiring — call the DataOS-Runtime composer (agent-neutral) and get the two
prompt sections (environment ops + task definition) rendered from the typed contract.

DataOS imports the runtime's ``compose`` directly (design decision A: sys.path +
import; the runtime composer is pure jinja2). Build the compose run-config from our
AutoRunConfig, then resolve() + render(). See DataOS/docs/runtime-contract.md.
"""
from __future__ import annotations

import sys
from functools import lru_cache

from .run_config import runtime_dir


@lru_cache(maxsize=1)
def _compose():
    rd = runtime_dir()
    if rd is None:
        raise RuntimeError(
            "DataOS-Runtime checkout not found — set DATAOS_RUNTIME_DIR or place it as ../DataOS-Runtime"
        )
    cdir = str(rd / "compose")
    if cdir not in sys.path:
        sys.path.insert(0, cdir)
    import compose  # noqa: E402  (from the runtime checkout)

    return compose


def build_run_config(cfg, *, parent_model: str, eval_limit: int) -> dict:
    """Assemble the run-config dict the composer expects, from our AutoRunConfig."""
    rt = getattr(cfg, "runtime_config", {}) or {}
    cluster = rt.get("cluster", {})
    paths = rt.get("paths", {})
    gres = str(cluster.get("gres", "gpu:1"))
    try:
        gpus = int(gres.rsplit(":", 1)[-1])
    except ValueError:
        gpus = 1
    return {
        "environment": rt.get("environment", "h100_slurm"),
        "task": rt.get("task", "posttrainbench"),
        "runtime_ref": rt.get("runtime_ref", "main"),
        "base_model": parent_model,
        "bench": cfg.target_bench,
        "resource": {
            "node": cluster.get("nodelist"),
            "gpus": gpus,
            "cpus": cluster.get("cpus_per_task"),
        },
        "eval_limit": eval_limit,
        "fixed_pool": paths.get("fixed_pool"),
        "benchmarks": paths.get("benchmarks"),
    }


def render_sections(cfg, *, parent_model: str, eval_limit: int) -> tuple[str, str]:
    """Return (env_ops, task_def) prompt text rendered from the runtime contract."""
    compose = _compose()
    rc = build_run_config(cfg, parent_model=parent_model, eval_limit=eval_limit)
    resolved = compose.resolve(rc)
    return compose.render(resolved)
