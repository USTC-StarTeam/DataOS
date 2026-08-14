#!/usr/bin/env python3
"""Sync the repo-owned tools in the H100 venvs to the versions DECLARED in env.yaml.

For every `tools.<env>.dataos_tools` entry in configs/environments/<env>/env.yaml:
check the installed dist version on the cluster (ssh) and, on mismatch, install the
package from the deploy checkout (tools/<name>/ — the checkout's pyproject version is
pinned equal to the declaration by tests/agent/test_env_tools_versions.py).

Nothing else ever mutates those venvs: version upgrades are a manual bump + this script.

Usage: python scripts/sync_envs.py [--environment PTB_H100_slurm] \
           [--deploy-root /data/hcgu/users/tjh/dataos_agentver] [--check]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _ssh(host: str, cmd: str) -> str:
    r = subprocess.run(
        ["ssh", host, cmd], capture_output=True, text=True, timeout=300, check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ssh {host} failed: {r.stderr.strip()[:500]}")
    return r.stdout.strip()


#: A tool's source dir. `tools/` holds the agent's instruments; dataos-operator is the
#: platform's own library and lives in `src/operator`. Searched rather than mapped, so a
#: future move needs no edit here.
#: Where a declared tool's source lives, tried in order. `dataos-tools` IS `tools/`;
#: `operator` is `src/operator`. Searched rather than mapped, so the next move costs no edit.
_PKG_BASES = ("", "tools", "src")


def _pkg_dir(tool: str) -> str:
    for base in _PKG_BASES:
        if (REPO / base / tool / "pyproject.toml").is_file():
            return f"{base}/{tool}"
    raise SystemExit(f"no pyproject.toml for tool {tool!r} under {' or '.join(_PKG_BASES)}/")


def _installed(host: str, venv: str, dist: str) -> str | None:
    out = _ssh(
        host,
        f"{venv}/bin/python -c \"import importlib.metadata as m;"
        f"print(m.version('dataos-{dist}'))\" 2>/dev/null || true",
    )
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--environment", default="PTB_H100_slurm")
    ap.add_argument("--deploy-root", default="/data/hcgu/users/tjh/dataos_agentver")
    ap.add_argument("--ssh-host", default="h100")
    ap.add_argument("--check", action="store_true", help="report only, never install")
    args = ap.parse_args()

    env = yaml.safe_load(
        (REPO / "configs" / "environments" / args.environment / "env.yaml").read_text()
    )
    mismatched = failed = False
    for env_name, spec in (env.get("tools") or {}).items():
        declared = (spec or {}).get("dataos_tools") or {}
        if not declared:
            continue
        venv = spec["path"]
        for tool, want in declared.items():
            have = _installed(args.ssh_host, venv, tool)
            if have == want:
                print(f"[ok] {env_name}: dataos-{tool} == {want}")
                continue
            mismatched = True
            print(f"[MISMATCH] {env_name}: dataos-{tool} installed={have} declared={want}")
            if args.check:
                continue
            src = f"{args.deploy_root}/{_pkg_dir(tool)}"
            print(f"  installing {src} -> {venv}")
            _ssh(
                args.ssh_host,
                f"$HOME/.local/bin/uv pip install --python {venv}/bin/python -q {src}",
            )
            have = _installed(args.ssh_host, venv, tool)
            if have != want:
                failed = True
                print(f"  [FAIL] after install: {have} != {want}")
            else:
                print(f"  [ok] now {have}")
    if failed:
        return 2
    return 1 if (args.check and mismatched) else 0


if __name__ == "__main__":
    sys.exit(main())
