"""Official iteration lifecycle for tasks whose contract selects an iteration budget."""

from __future__ import annotations

import json
from pathlib import Path

from . import runtime, runtime_compose


def _policy(cfg) -> dict:
    return dict((runtime_compose.task_agent_config(cfg).get("budget") or {}))


def _command(cfg, action: str, *, expected_iteration: int | None = None) -> list[str] | None:
    policy = _policy(cfg)
    if policy.get("unit") != "iterations":
        return None
    coordinator = str(policy.get("coordinator") or "")
    if not coordinator:
        raise ValueError("iteration-budget task is missing agent.budget.coordinator")
    max_rounds = cfg.budget.max_rounds
    if not isinstance(max_rounds, int) or max_rounds < 1:
        raise ValueError("iteration-budget task requires a positive max_rounds")

    run = runtime_compose.build_run_config(
        cfg, parent_model=cfg.h100_base_model or "base", eval_limit=-1
    )
    resolved = runtime_compose.resolve(run)
    task = resolved["task"]
    data_stage = task["stages"]["data"]
    tool = str(data_stage.get("tool") or "")
    providers = [
        value
        for value in resolved["environment"]["tools"].values()
        if isinstance(value, dict) and tool in (value.get("provides") or [])
    ]
    if len(providers) != 1:
        raise ValueError(f"iteration coordinator requires exactly one provider for {tool!r}")
    provider = providers[0]
    paths = (cfg.runtime_config or {}).get("paths") or {}
    deploy_root = str(paths.get("deploy_root") or "")
    runs_base = str(paths.get("runs_base") or "")
    task_name = str((cfg.runtime_config or {}).get("task") or "")
    command = [
        str(Path(str(provider["path"])) / "bin" / "python"),
        str(Path(deploy_root) / "configs" / "tasks" / task_name / coordinator),
        action,
        "--project-root",
        str(Path(runs_base) / cfg.project_id / "_curation_suite"),
        "--profile",
        str(provider["profile"]),
        "--task",
        str(cfg.target_bench),
        "--iterations",
        str(max_rounds),
    ]
    if expected_iteration is not None:
        command.extend(["--expected-iteration", str(expected_iteration)])
    return command


def claim(cfg, node: dict) -> dict | None:
    """Claim or resume the official target corresponding to this durable DataOS node."""
    expected = int(node["round"]) + 1
    if (
        node.get("iteration") == expected
        and node.get("suite_dir")
        and node.get("suite_target")
        and node.get("suite_run_dir")
        and node.get("submission_path")
    ):
        return {
            "status": "target_activated",
            "iteration": expected,
            "suite_dir": node["suite_dir"],
            "target_key": node["suite_target"],
            "target_state": node.get("suite_target_state") or "active",
            "run_dir": node["suite_run_dir"],
            "default_submission_path": node["submission_path"],
        }
    command = _command(cfg, "claim", expected_iteration=expected)
    if command is None:
        return None
    raw = runtime.exec_host_command(cfg, command)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"iteration coordinator returned invalid JSON: {raw[-2000:]}") from exc
    if result.get("status") != "target_activated" or result.get("iteration") != expected:
        raise RuntimeError(
            f"iteration coordinator returned the wrong target for round {node['round']}: {result}"
        )
    return result


def finalize(cfg) -> dict | None:
    """Close the official suite after DataOS consumes its configured iteration budget."""
    command = _command(cfg, "finalize")
    if command is None:
        return None
    raw = runtime.exec_host_command(cfg, command)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"iteration finalizer returned invalid JSON: {raw[-2000:]}") from exc
    if result.get("status") != "suite_completed":
        raise RuntimeError(f"iteration suite did not finalize: {result}")
    return result
