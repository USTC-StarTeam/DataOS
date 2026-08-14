"""A kind of work, and the one path from intent to a running job.

Three objects, three sources, and each is built from exactly one of them:

===========  =====================================  ====================================
object       built from                             answers
===========  =====================================  ====================================
``Spec``     the run's intent                       what work to do
``Stage``    the environment (``env.yaml``)         how this deployment does that work
``Launcher`` the environment + the run's choices    where it runs
===========  =====================================  ====================================

The middle row is why ``Stage`` takes ``env`` in its constructor rather than in ``build``. Where
a venv lives, which bridge script to call, what the frozen eval protocol declares — those are
facts about the machine, not about the experiment, and a stage that reads them from a Spec is a
stage whose Spec has to carry them. That is how ``TrainSpec`` came to hold a ``cluster`` block.

``submit`` is here, once, because it was written three times: each CLI validated (or forgot to),
built a Job, resolved a launcher, submitted, and wrote a record — with its own bugs in its own
copy. ``prepare`` is the funnel both ``build`` and ``submit`` go through; dataos-train's submit
used to skip validation, so a spec its backend rejected at build time was submitted anyway.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import yaml

from .placement import launcher_from_config, launcher_name, placement
from .registry import get_launcher
from .spec import Spec, SpecError
from .types import Handle, Job, Status


class Stage(ABC):
    """One kind of work — data, train, eval — as this deployment performs it."""

    #: "data" | "train" | "eval"
    name: ClassVar[str]

    def __init__(self, env: dict) -> None:
        self.env = env

    @staticmethod
    def load_env(repo: str | Path, environment: str) -> dict:
        """The deployment contract, off the checkout. One reader; there were three."""
        path = Path(repo) / "configs" / "environments" / environment / "env.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    @classmethod
    def from_repo(cls, repo: str | Path, environment: str, **kw: Any) -> Stage:
        return cls(cls.load_env(repo, environment), **kw)

    def prepare(self, spec: Spec) -> Job:
        """Validate, then build. Raises :class:`SpecError` with every reason at once."""
        errors = self.validate(spec)
        if errors:
            raise SpecError(errors)
        return self.build(spec)

    def submit(self, spec: Spec, resource: dict | None = None) -> Handle:
        """Intent in, running job out. ``resource`` is the run's placement choices."""
        job = self.prepare(spec)
        return launcher_from_config(self.env, resource).submit(job)

    def preview(self, spec: Spec, resource: dict | None = None) -> dict[str, Any]:
        """Everything ``submit`` would do, as JSON. What ``--dry-run`` prints.

        Placement is in here even though it is no longer on the Job — a dry run answers "what
        gets submitted", and half that answer moved into the launcher.
        """
        return {
            "stage": self.name,
            "launcher": launcher_name(self.env),
            "placement": placement(self.env, resource),
            "spec": spec.model_dump(mode="json"),
            "job": self.prepare(spec).model_dump(mode="json"),
        }

    @staticmethod
    def status(handle: Handle) -> Status:
        """Poll. Needs no placement — the handle already says who took the job."""
        return get_launcher(handle.launcher).status(handle)

    @abstractmethod
    def validate(self, spec: Spec) -> list[str]:
        """Every reason this spec cannot run here, or an empty list."""

    @abstractmethod
    def build(self, spec: Spec) -> Job:
        """Render the spec into a launcher-agnostic Job."""
