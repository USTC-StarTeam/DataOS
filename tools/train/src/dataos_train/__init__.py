"""dataos-train: a unified multi-backend training layer for DataOS.

Agents emit a :class:`~dataos_train.spec.TrainSpec` and get back a
:class:`~dataos_train.types.TrainResult`; backends (LLaMA-Factory, verl) and launchers
(Slurm, bare-metal) sit behind that contract.
"""

from __future__ import annotations

from importlib import metadata as _metadata

from .registry import (
    backend_names,
    get_backend,
    get_launcher,
    register_backend,
    register_launcher,
)
from .spec import GRPO, SFT, DatasetRef, EnvSpec, OnlineOpSpec, TrainSpec
from .types import (
    Capabilities,
    Handle,
    JobSpec,
    LaunchPlan,
    Resources,
    Status,
    TrainResult,
)

__all__ = [
    "TrainSpec",
    "SFT",
    "GRPO",
    "DatasetRef",
    "EnvSpec",
    "OnlineOpSpec",
    "JobSpec",
    "LaunchPlan",
    "Resources",
    "Handle",
    "Status",
    "TrainResult",
    "Capabilities",
    "get_backend",
    "get_launcher",
    "register_backend",
    "register_launcher",
    "backend_names",
]
# Read from the installed distribution, not a second hand-maintained literal: this had
# drifted to 0.1.0a0 while pyproject said 0.1.2. Same fix as dataos-operator/-eval.
__version__ = _metadata.version("dataos-train")
