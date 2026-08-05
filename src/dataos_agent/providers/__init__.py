"""Importing this package self-registers the built-in providers."""

from . import fake  # noqa: F401  — registers "fake"

# Real CLI backends self-register on import; added in phase 4:
for _m in ("claude", "kimi", "qwen", "codex"):
    try:
        __import__(f"{__name__}.{_m}")
    except ImportError:
        pass  # not written yet / optional
