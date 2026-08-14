"""The training stage: one stage, several backends.

Training is the one stage whose spec an agent AUTHORS — the method is what DataOS searches, so
``TrainSpec`` closes over a method schema and ``dataos-train validate`` is a real gate. It is
also the one stage with a plugin point *below* the stage: LLaMA-Factory and verl build the same
spec into different jobs, so :class:`~dataos_train.protocols.Backend` sits under ``TrainStage``
the way ``TrainStage`` sits under a launcher.

    Spec  →  Stage  →  Backend  →  Job  →  Launcher
    what     which     how         run      where
    to do    stage     this        it       it runs
                       backend
                       does it

Nothing here decides where the job runs. That was not true until 2026-08-14: ``TrainSpec``
carried a ``cluster`` block, filled by the brain from ``env.yaml``, and it reached the launcher
by riding on the Job. It was the third hand-written copy of a rule the other two instruments
already read from config, and it went stale in exactly the way a third copy does.
"""

from __future__ import annotations

from dataos_launch.stage import Stage
from dataos_launch.types import Job

from .registry import get_backend
from .spec import TrainSpec


class TrainStage(Stage):
    """Trains a model, via whichever backend the spec names."""

    name = "train"

    def validate(self, spec: TrainSpec) -> list[str]:
        from .precheck import precheck_disk

        return get_backend(spec.backend).validate(spec) + precheck_disk(spec)

    def build(self, spec: TrainSpec) -> Job:
        return get_backend(spec.backend).build(spec)
