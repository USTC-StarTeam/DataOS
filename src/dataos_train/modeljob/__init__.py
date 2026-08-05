"""ModelJob — score data with a model, no training.

The layer-2 execution slot of the DataOS operator stack (model × data, one forward
pass). Peer of the ``TrainSpec`` lifecycle with the same CLI shape; emits
``scores.jsonl`` instead of a checkpoint. The package is torch-free — torch enters only
inside :mod:`dataos_train.modeljob.score_entry` at job time.
"""

from __future__ import annotations

from .backend import ScoreResult, build, collect, validate
from .spec import GradientInteraction, Interaction, LossInteraction, ModelJobSpec

__all__ = [
    "GradientInteraction",
    "Interaction",
    "LossInteraction",
    "ModelJobSpec",
    "ScoreResult",
    "build",
    "collect",
    "validate",
]
