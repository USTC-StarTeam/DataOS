"""Shared backend helpers."""

from __future__ import annotations

from pathlib import Path

_HF_WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin", "*.bin")


def is_hf_model_dir(path: str | Path) -> bool:
    """True if ``path`` looks like a complete HF model dir (config + weights)."""
    p = Path(path)
    if not (p / "config.json").is_file():
        return False
    return any(any(p.glob(pat)) for pat in _HF_WEIGHT_GLOBS)


def tail_text(path: str | Path, n: int = 50) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])
