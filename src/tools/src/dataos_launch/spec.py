"""What a stage intends, before any deployment is resolved into it.

The first of the three nouns (see :mod:`dataos_launch.types` for the other two). A ``Spec`` is
the run's intent — this data, this method, this bench — and nothing else. It says what work to
do, never where it runs: that is placement, and it belongs to the launcher.

The three stages are peers, and this is what makes them so. They did not look like peers before:
``TrainSpec`` was a closed pydantic schema an agent writes by hand, while evaluation and data
building had no spec at all — each CLI turned an ``argparse.Namespace`` straight into a ``Job``,
which is why the same four flags were parsed, defaulted and forwarded three times over. The
difference between the stages is real but it is narrower than that: what an agent AUTHORS
differs (the training method; the data recipe; for evaluation, nothing at all — the protocol is
frozen), while what the run *intends* has the same shape everywhere.

So ``EvalSpec`` and ``DataSpec`` are constructed by their tool rather than written by an agent.
That is not a lesser kind of spec; it is the same noun with a different author.
"""

from __future__ import annotations

import json
from abc import ABC
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from .model import Model
from .types import Resources


class SpecError(ValueError):
    """A spec that cannot be built. Carries every reason, not just the first."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class Spec(Model, ABC):
    """The run's intent for one stage.

    ``resources`` is what the program NEEDS (see :class:`~dataos_launch.types.Resources`); the
    run's placement CHOICES — which QOS, which node — are deliberately not here. Both arrive
    from the same ``run.resource`` block in the run config, which is why they were confused for
    one thing: ``gpus``/``cpus`` describe the work and travel with the spec, ``qos``/``node``
    describe the machine and travel to the launcher.
    """

    #: which stage this is; also names the file :meth:`write` produces
    stage: ClassVar[str]

    name: str
    run_dir: str
    resources: Resources = Field(default_factory=Resources)

    @property
    def spec_path(self) -> Path:
        """``<run_dir>/specs/<stage>-<name>.json`` — one file per SPEC, not one per stage.

        A run_dir holds a node, and a node submits many jobs: one data build, one training run,
        then one evaluation per checkpoint. Writing ``<stage>_spec.json`` gave all six evals the
        same path, so the file that claims to record what was intended held only the last of
        them — and the experimenter took to inventing ``eval_handle_final.json`` and
        ``eval_handle_ck100.json`` to keep the others, which is the recipe naming its own
        manifest all over again.
        """
        return Path(self.run_dir) / "specs" / f"{self.stage}-{self.name}.json"

    def write(self) -> Path:
        """Persist next to the run. What was intended, readable long after the job is gone."""
        path = self.spec_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path):
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
