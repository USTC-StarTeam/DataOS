"""Provenance capture for reproducibility manifests — versions, hardware, git, image.

Every collector is **lazy and failure-safe**: a missing package / no GPU / not-a-git-repo
yields ``None`` (or a partial dict) instead of raising, so recording provenance never breaks
a run. Heavy deps (torch) are imported only inside :func:`hardware`. :func:`package_versions`
and :func:`hardware` are cached — they don't change within a process.

What this pins (the *recorded* half of reproducibility; the *enforced* half is the image):
versions of the three engines (dataos_operator / llamafactory / lm-eval) + their deps, the GPU
/ CUDA / host (numerics depend on hardware — recorded so drift is explainable, not preventable),
the DataOS git sha, and the container ``image@digest`` if the runtime set ``DATAOS_IMAGE``.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

# Packages whose version determines an SFT train/eval run's behavior.
_TRACKED = (
    "dataos",
    "dataos-operator",
    "datasets",
    "pyarrow",
    "torch",
    "transformers",
    "accelerate",
    "llamafactory",
    "lm-eval",
)


def _version(dist: str) -> str | None:
    """importlib version, tolerant of hyphen/underscore naming; None if absent."""
    for name in (dist, dist.replace("-", "_"), dist.replace("_", "-")):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            return None
    return None


@lru_cache(maxsize=1)
def package_versions() -> dict[str, Any]:
    """Versions of the packages that determine an experiment's behavior (None if absent)."""
    versions: dict[str, Any] = {dist: _version(dist) for dist in _TRACKED}
    versions["python"] = platform.python_version()
    # For the running DataOS, the source __version__ is the truth (editable installs leave
    # importlib metadata stale until re-sync); prefer it for the "dataos" entry.
    try:
        import dataos  # noqa: PLC0415

        versions["dataos"] = getattr(dataos, "__version__", None) or versions.get("dataos")
    except Exception:
        pass
    return versions


@lru_cache(maxsize=1)
def hardware() -> dict[str, Any]:
    """GPU / CUDA / host info (lazy torch; gpu fields stay None when torch or GPU absent)."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu_name": None,
        "gpu_count": 0,
        "cuda": None,
    }
    try:
        import torch  # noqa: PLC0415 - heavy, intentionally lazy

        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def git_sha(path: str | Path) -> dict[str, Any] | None:
    """``{"sha", "dirty"}`` for the git repo containing ``path``; None if not a repo / no git."""
    target = Path(path)
    cwd = str(target if target.is_dir() else target.parent)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        if head.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return {"sha": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except Exception:
        return None


def image_ref() -> str | None:
    """The container ``image@digest`` this runs in, if the runtime exported ``DATAOS_IMAGE``."""
    return os.environ.get("DATAOS_IMAGE") or None
