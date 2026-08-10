"""Each CLI's own *context file* name — the file it reads as standing instructions.

    AGENTS.md (codex, kimi) · CLAUDE.md (claude) · QWEN.md (qwen)

:func:`context_filename` only reports the name. Building that file is the caller's job, because
its content is the caller's, not this package's: point ``workspace`` at a directory you control,
write the file there, and the CLI picks it up.

WHY THERE IS NO CONFIG-ROOT ISOLATION HERE
------------------------------------------
This module used to relocate each CLI's whole config root (``CODEX_HOME`` /
``CLAUDE_CONFIG_DIR`` / ``KIMI_CODE_HOME`` / ``QWEN_HOME``) into a private directory per run, so a
run would ignore the machine's config, skills, hooks, MCP servers and global context file. That is
incompatible with how these CLIs authenticate, and the incompatibility is not fixable from here:

* Their OAuth credential lives inside the config root and carries a SINGLE-USE, rotating refresh
  token. Give N processes their own copy of it and the first refresh invalidates the token for
  every other copy — and for the real file too. Seven concurrent runs did exactly that on
  2026-08-10 and left ``codex`` needing a fresh interactive login.
* Symlinking the credential instead is not a fix either: with a config root that otherwise looks
  empty, a CLI may RE-INITIALISE its credential store rather than refresh it, and the write lands
  on the real file through the link. ``kimi`` did that, zeroing a live login.

So the machine's config root is shared, as it was before, and isolation is done the one way that
has no such coupling: the run's *cwd* is a private directory outside the repo (see
``dataos.agent.llm.config.role_cwd``), which is what keeps a CLI from auto-discovering the repo's
development-time AGENTS.md. Anything global in ``~/.codex`` / ``~/.claude`` is shared with the
host on purpose — do not re-add a relocation without solving the credential problem first.
"""

from __future__ import annotations

#: provider -> the name of the context file that provider reads
_CONTEXT_FILE = {
    "claude": "CLAUDE.md",
    "codex": "AGENTS.md",
    "kimi": "AGENTS.md",
    "qwen": "QWEN.md",
}


def context_filename(provider: str) -> str | None:
    """The name of the CLI's context file, or None for a provider that has none."""
    return _CONTEXT_FILE.get(provider)
