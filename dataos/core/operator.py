from __future__ import annotations

from abc import ABC, abstractmethod

from dataos.core.artifact import Artifact
from dataos.core.sandbox import Sandbox


class Operator(ABC):
    """Abstract operator executed by the fixed runtime."""

    name = ""
    description = ""

    @abstractmethod
    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> Artifact | list[Artifact]:
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}
