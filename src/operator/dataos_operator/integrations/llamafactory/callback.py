"""LLaMA-Factory callback glue for DataOS policy refresh and snapshots."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback


class DataOSCallback(TrainerCallback):
    """Forward trainer lifecycle events to a DataOS training controller."""

    def __init__(self, controller: Any) -> None:
        self.controller = controller

    def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        self.controller.sampler_epoch += 1

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        self.controller.refresh_policy(step=int(state.global_step))

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, control, kwargs
        self.controller.save_snapshot(step=int(state.global_step))

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        del args, state, control, kwargs
        self.controller.finalize()
