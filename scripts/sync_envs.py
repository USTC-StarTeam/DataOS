#!/usr/bin/env python3
"""Sync the repo-owned tools in an environment's venvs to the versions DECLARED in env.yaml.

For every `tools.<env>.dataos_tools` entry in configs/environments/<env>/env.yaml:
check the installed dist version on the exec host (ssh) and, on mismatch, install the
package from the deploy checkout (the checkout's pyproject version is pinned equal to
the declaration by tests/agent/test_env_tools_versions.py).

Nothing else ever mutates those venvs: version upgrades are a manual bump + this script.

The ssh host is read from the environment's own `access.host` — the environment already
says how to reach itself, and a second copy on the command line is a second thing to get
wrong. `--deploy-root` has no such source: where a checkout sits is a run-level fact
(run JSON), which env.yaml deliberately does not carry.

Usage: python scripts/sync_envs.py --environment <name> --deploy-root <path> [--check]
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


#: Where a declared tool's source lives, tried in order. Both are under `src/` now
#: (`src/dataos-tools`, `src/operator`); the bare "" and "tools" entries keep an older checkout
#: resolvable. Searched rather than mapped, so a move costs no edit here — which is why this
#: file needed none when `tools/` became `src/tools/` and then `src/dataos-tools/`.
_PKG_BASES = ("src", "", "tools")


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
    ap.add_argument("--environment", required=True)
    ap.add_argument("--deploy-root", required=True,
                    help="the deploy checkout on the exec host (run-level fact)")
    ap.add_argument("--ssh-host", default=None,
                    help="override; defaults to the environment's access.host")
    ap.add_argument("--check", action="store_true", help="report only, never install")
    args = ap.parse_args()

    env = yaml.safe_load(
        (REPO / "configs" / "environments" / args.environment / "env.yaml").read_text()
    )
    host = args.ssh_host or ((env.get("access") or {}).get("host") or "")
    if not host:
        raise SystemExit(
            f"{args.environment} declares no access.host — pass --ssh-host, or point at a "
            f"remote environment (a local one has no venvs to reach over ssh)."
        )
    mismatched = failed = False
    for env_name, spec in (env.get("tools") or {}).items():
        declared = (spec or {}).get("dataos_tools") or {}
        if not declared:
            continue
        venv = spec["path"]
        for tool, want in declared.items():
            have = _installed(host, venv, tool)
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
                host,
                f"$HOME/.local/bin/uv pip install --python {venv}/bin/python -q {src}",
            )
            have = _installed(host, venv, tool)
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
