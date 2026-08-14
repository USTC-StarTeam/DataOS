"""dataos-launch — run a rendered job somewhere, and read its progress back.

A ``Job`` says WHAT to run; a ``Launcher`` decides WHERE. Splitting the two is what lets
one training or evaluation job go to Slurm on a cluster or to a subprocess on a workstation without
either side knowing about the other. Extracted from dataos-train on 2026-08-13 so dataos-eval
could stop hardcoding ``sbatch``; the alternative — a second copy — is the shape of defect this
repository keeps paying for (two producers, two same-named helpers, two truths).
"""

from __future__ import annotations

from importlib import metadata as _metadata

from .launchers.base import Launcher
from .placement import launcher_from_config, launcher_name, placement
from .registry import get_launcher, launcher_names, register_launcher
from .types import Handle, Job, LaunchPlan, Resources, Status

# One distribution now (dataos-tools): the launcher and the three instruments always ship
# together, so they share a version. Read from the installed dist rather than a literal —
# a second hand-maintained copy is how this constant once sat three minors stale.
__version__ = _metadata.version("dataos-tools")

__all__ = [
    "Handle",
    "Job",
    "LaunchPlan",
    "Launcher",
    "Resources",
    "Status",
    "get_launcher",
    "launcher_from_config",
    "launcher_name",
    "launcher_names",
    "placement",
    "register_launcher",
]
