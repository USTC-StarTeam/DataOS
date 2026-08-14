"""env.yaml `dataos_tools` declarations must equal the in-repo tool versions.

The H100 venvs install each tool from the deploy checkout; the declared version in
env.yaml is the ledger of what those venvs contain (sync_envs.py enforces it on the
cluster). This test pins declaration == the tool's pyproject version, so bumping a
tool without updating the declaration (or vice versa) fails here."""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
#: `tools/` holds the agent's instruments; dataos-operator is the platform's own operator
#: library and lives in `src/operator`. Both are workspace members with their own version.
_PKG_BASES = ("", "tools", "src")


def _pyproject(tool: str) -> Path:
    for base in _PKG_BASES:
        p = REPO / base / tool / "pyproject.toml"
        if p.is_file():
            return p
    raise AssertionError(
        f"declared tool {tool!r} has no pyproject.toml under {' or '.join(_PKG_BASES)}/"
    )


def _tool_version(tool: str) -> str:
    text = _pyproject(tool).read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m, f"{_pyproject(tool)} has no version"
    return m.group(1)


def test_dataos_tools_declarations_match_tool_versions():
    envs_dir = REPO / "configs" / "environments"
    checked = 0
    for env_yaml in sorted(envs_dir.glob("*/env.yaml")):
        env = yaml.safe_load(env_yaml.read_text(encoding="utf-8"))
        for tool_env, spec in (env.get("tools") or {}).items():
            for tool, declared in ((spec or {}).get("dataos_tools") or {}).items():
                actual = _tool_version(tool)
                assert declared == actual, (
                    f"{env_yaml.parent.name}: tools.{tool_env}.dataos_tools.{tool}="
                    f"{declared} but tools/{tool}/pyproject.toml says {actual} — bump "
                    "both together, then run scripts/sync_envs.py on the cluster"
                )
                checked += 1
    assert checked, "no dataos_tools declarations found — the ledger must not be empty"
