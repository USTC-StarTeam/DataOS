"""dataos-launch — run a rendered job somewhere, and read its progress back.

A ``JobSpec`` says WHAT to run; a ``Launcher`` decides WHERE. Splitting the two is what lets
one training or evaluation job go to Slurm on a cluster or to a subprocess on a workstation without
either side knowing about the other. Extracted from dataos-train on 2026-08-13 so dataos-eval
could stop hardcoding ``sbatch``; the alternative — a second copy — is the shape of defect this
repository keeps paying for (two producers, two same-named helpers, two truths).
"""

from __future__ import annotations

from importlib import metadata as _metadata

from .protocols import Launcher
from .registry import get_launcher, launcher_names, register_launcher
from .types import Handle, JobSpec, LaunchPlan, RayPlan, Resources, SlurmExtras, Status

__version__ = _metadata.version("dataos-launch")

__all__ = [
    "Handle",
    "JobSpec",
    "LaunchPlan",
    "Launcher",
    "RayPlan",
    "Resources",
    "SlurmExtras",
    "Status",
    "get_launcher",
    "launcher_names",
    "register_launcher",
]
