"""Isolated config roots — point a CLI at a private config dir instead of the
machine's (``~/.codex``, ``~/.claude``, ...), so a run is unaffected by whatever
host config, skills, hooks, MCP servers or global context file happen to exist
there. Every CLI here reads its whole configuration (including its own sessions
and memories) out of one directory, so relocating that directory isolates a run
without disabling anything.

This is also the one place that knows each CLI's *global context file* name.
:func:`context_filename` only reports it — building that file is the caller's
job, because its content is the caller's, not this package's.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# provider -> (config-root env var, global context filename, what to seed the root with)
_SPEC: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "claude": ("CLAUDE_CONFIG_DIR", "CLAUDE.md", ("~/.claude/.credentials.json",)),
    "codex": ("CODEX_HOME", "AGENTS.md", ("~/.codex/auth.json",
                                          "~/.codex/models_cache.json")),
    # kimi keeps its endpoint/model config in the same file as its key, so config.toml
    # has to come along — kimi/qwen are isolated from context files and state, not config
    "kimi": ("KIMI_CODE_HOME", "AGENTS.md", ("~/.kimi-code/credentials",
                                             "~/.kimi-code/config.toml")),
    "qwen": ("QWEN_HOME", "QWEN.md", ("~/.qwen/settings.json",)),
}


def context_filename(provider: str) -> str | None:
    """The name of the CLI's global context file (``AGENTS.md`` / ``CLAUDE.md`` /
    ``QWEN.md``), or None for a provider that has none. Callers write that file into
    the run's workspace themselves; this package never decides its content."""
    spec = _SPEC.get(provider)
    return spec[1] if spec else None


def _seed(src: Path, dst: Path) -> None:
    """Place ``src`` into the isolated root — always a COPY, never a symlink.

    A symlink is actively dangerous here: given a config root with no auth state, these
    CLIs may RE-INITIALIZE their credential store instead of refreshing it, and through
    a symlink that rewrite lands on the user's real file and destroys their login (kimi
    does exactly this). The copy is re-seeded only when it is missing, or when the real
    file is newer (the user re-authenticated) — never mid-run, so a token the CLI
    refreshed inside the isolated root survives."""
    if not src.exists():
        return
    if src.is_dir():
        if not dst.exists():
            shutil.copytree(src, dst)
        return
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    shutil.copy2(src, dst)  # copy2 preserves the 0600 mode of a credential file


def home_env(provider: str, home: str) -> dict[str, str]:
    """Prepare ``home`` as an isolated config root for ``provider``, and return the env
    that points the CLI at it.

    A provider with no known config root yields ``{}`` (no isolation). Note that the root
    ends up holding a COPY of the credential: if its refresh token ever expires, delete
    the root and it re-seeds from the real one."""
    spec = _SPEC.get(provider)
    if spec is None:
        return {}
    var, _, seeds = spec
    root = Path(home)
    root.mkdir(parents=True, exist_ok=True)
    for src in seeds:
        s = Path(src).expanduser()
        _seed(s, root / s.name)
    return {var: str(root)}
