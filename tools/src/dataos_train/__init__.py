"""dataos-train: a unified multi-backend training layer for DataOS.

Agents emit a :class:`~dataos_train.spec.TrainSpec` and get back a
:class:`~dataos_train.types.TrainResult`; backends (LLaMA-Factory, verl) sit behind that
contract. WHERE the resulting job runs is not this package's concern — a backend renders a
launcher-agnostic ``JobSpec`` and :mod:`dataos_launch` runs it, on Slurm or bare metal.
"""

from __future__ import annotations

from importlib import metadata as _metadata

from .registry import backend_names, get_backend, register_backend
from .spec import GRPO, SFT, DatasetRef, EnvSpec, OnlineOpSpec, TrainSpec
from .types import Capabilities, TrainResult

__all__ = [
    "GRPO",
    "SFT",
    "Capabilities",
    "DatasetRef",
    "EnvSpec",
    "OnlineOpSpec",
    "TrainResult",
    "TrainSpec",
    "backend_names",
    "get_backend",
    "register_backend",
]
# Read from the installed distribution, not a second hand-maintained literal: this had
# drifted to 0.1.0a0 while pyproject said 0.1.2. Same fix as dataos-operator/-eval.
# One distribution now (dataos-tools): the launcher and the three instruments always ship
# together, so they share a version. Read from the installed dist rather than a literal —
# a second hand-maintained copy is how this constant once sat three minors stale.
__version__ = _metadata.version("dataos-tools")
