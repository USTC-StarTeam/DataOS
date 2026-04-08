from __future__ import annotations

from dataos.core.artifact import Artifact
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.train import run_train_stage


class TrainLlamaFactoryOperator(Operator):
    """Compatibility wrapper around the runtime training functions."""

    name = "train_llamafactory"
    description = "Compatibility wrapper that delegates training to the runtime functions."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs):
        return run_train_stage(inputs, sandbox=sandbox, **kwargs)
