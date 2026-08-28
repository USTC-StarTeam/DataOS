"""Load the stage contracts selected by a task.

The root ``task.yaml`` describes the problem and owns the data-stage contract.  Train and eval
each live in their own ``task.yaml`` because those two instruments consume them independently.
The root file links them explicitly; directory existence is never a behavior switch.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import yaml

STAGES = ("data", "train", "eval")
OWNERS = ("agent", "frozen")


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"task contract does not exist: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"task contract must be a mapping: {path}")
    return data


def _relative_contract(task_dir: Path, value: object, *, stage: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"task.yaml.stages.{stage}.contract must be a non-empty path")
    task_root = task_dir.resolve()
    path = (task_dir / value).resolve()
    if path != task_root and task_root not in path.parents:
        raise ValueError(f"task.yaml.stages.{stage}.contract escapes the task directory: {value}")
    return path


def load_task_contract(repo: str | Path, task_config: str) -> dict[str, Any]:
    """Load the root task contract and its explicit data/train/eval stage contracts."""
    task_dir = Path(repo) / "configs" / "tasks" / task_config
    root = _read_mapping(task_dir / "task.yaml")
    if root.get("name") != task_config:
        raise ValueError(
            f"task contract name {root.get('name')!r} does not match directory {task_config!r}"
        )
    declared = root.get("stages")
    if not isinstance(declared, dict):
        raise ValueError("task.yaml.stages must explicitly declare data, train, and eval")

    resolved: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        entry = declared.get(stage)
        if not isinstance(entry, dict):
            raise ValueError(f"task.yaml.stages.{stage} must be a mapping")
        if "contract" in entry:
            if set(entry) != {"contract"}:
                raise ValueError(
                    f"task.yaml.stages.{stage} must contain only contract when linking a file"
                )
            contract = _read_mapping(_relative_contract(task_dir, entry["contract"], stage=stage))
        else:
            contract = dict(entry)
        if contract.get("stage") != stage:
            raise ValueError(f"{stage} contract must declare stage: {stage}")
        driver = contract.get("driver")
        if not isinstance(driver, str) or not driver.strip():
            raise ValueError(f"{stage} contract must declare a non-empty driver")
        if driver == "external_cli" and not contract.get("tool"):
            raise ValueError(f"{stage} external_cli contract must declare tool")
        if contract.get("ownership") not in OWNERS:
            raise ValueError(
                f"{stage} contract ownership must be one of {', '.join(OWNERS)}"
            )
        resolved[stage] = contract

    out = dict(root)
    out["stages"] = resolved
    out["assets_root"] = str(task_dir)
    return out


def load_stage_contract(
    repo: str | Path, task_config: str, stage: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(root task, selected stage)`` for one instrument."""
    if stage not in STAGES:
        raise ValueError(f"unknown task stage {stage!r}; expected one of {', '.join(STAGES)}")
    task = load_task_contract(repo, task_config)
    return task, task["stages"][stage]


def tool_provider(env: dict, capability: str) -> dict[str, Any]:
    """Return the one environment tool provider for a declared capability."""
    matches = [
        value
        for value in (env.get("tools") or {}).values()
        if isinstance(value, dict) and capability in (value.get("provides") or [])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"environment must declare exactly one tools provider for {capability!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def validate_provider_pin(provider: dict, capability: str) -> list[str]:
    """Verify a provider's declared package version and checkout revision, when pinned."""
    errors: list[str] = []
    executable = Path(str(provider.get("path"))) / "bin" / capability
    if not executable.is_file():
        errors.append(f"configured external tool does not exist: {executable}")
    version = provider.get("version")
    distribution = provider.get("distribution")
    if version or distribution:
        if not (version and distribution):
            errors.append("external tool pin requires both distribution and version")
        else:
            python = Path(str(provider.get("path"))) / "bin" / "python"
            probe = (
                subprocess.run(
                    [
                        str(python),
                        "-c",
                        "import importlib.metadata as m,sys;"
                        "sys.stdout.write(m.version(sys.argv[1]))",
                        str(distribution),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if python.is_file()
                else None
            )
            if probe is None or probe.returncode != 0:
                errors.append(f"cannot read installed version for {distribution!r} from {python}")
            elif probe.stdout.strip() != str(version):
                errors.append(
                    f"configured {distribution} version is {version}, installed is "
                    f"{probe.stdout.strip()}"
                )
    revision = provider.get("revision")
    root = provider.get("root")
    if revision or root:
        if not (revision and root):
            errors.append("external tool checkout pin requires both root and revision")
        else:
            probe = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode != 0:
                errors.append(f"cannot read external tool revision at {root}")
            elif probe.stdout.strip() != str(revision):
                errors.append(
                    f"configured checkout revision is {revision}, installed is "
                    f"{probe.stdout.strip()}"
                )
    return errors
