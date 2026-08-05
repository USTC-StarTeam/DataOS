"""The contained LLaMA-Factory trainer patch ("魔改") that wires online operators.

Runs ONLY inside the training job (needs torch + llamafactory) — never imported at package
import time. Three integration points:
  - ``reweight`` hooks: per-sample loss reweighting inside ``compute_loss``.
  - ``step`` hooks: a ``TrainerCallback`` firing each optimizer step.
  - ``select`` hooks: ``observe(batch)`` inside ``compute_loss`` + ``decide(ctx)`` at epoch
    boundaries, applied through a :class:`MutableSubsetSampler` wrapping the train sampler.

The LF trainer class path is version-sensitive; ``entry.py`` monkeypatches the symbol the
SFT workflow instantiates, so a version bump touches one line there.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch
from transformers import TrainerCallback

from ....online import MutableSubsetSampler, apply_reweight

# Set by entry.py before LF builds the trainer (it instantiates the class with its own args).
_ACTIVE_HOOKS: list[Any] = []


def set_active_hooks(hooks: list[Any]) -> None:
    global _ACTIVE_HOOKS
    _ACTIVE_HOOKS = list(hooks)


class StepHookCallback(TrainerCallback):
    def __init__(self, hooks: list[Any]) -> None:
        self.hooks = [h for h in hooks if getattr(h, "hook_point", "") == "step"]

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        for hook in self.hooks:
            with suppress(Exception):
                hook({"step": state.global_step, "metrics": getattr(state, "log_history", [])})
        return control


class SelectHookCallback(TrainerCallback):
    """Applies ``select`` hook decisions at epoch boundaries to the subset sampler."""

    def __init__(self, hooks: list[Any], sampler_ref: dict[str, Any]) -> None:
        self.hooks = hooks
        self.sampler_ref = sampler_ref  # late-bound: the sampler is built after __init__

    def on_epoch_end(self, args, state, control, **kwargs):  # noqa: ANN001
        sampler: MutableSubsetSampler | None = self.sampler_ref.get("sampler")
        if sampler is None:
            return control
        ctx = {
            "epoch": int(getattr(state, "epoch", None) or 0),
            "step": state.global_step,
            "n_rows": self.sampler_ref.get("n_rows"),
        }
        for hook in self.hooks:
            decide = getattr(hook, "decide", None)
            if decide is None:
                continue
            with suppress(Exception):
                decision = decide(ctx)
                if isinstance(decision, dict) and decision.get("indices") is not None:
                    sampler.set_allowed(decision["indices"])
        return control


class _SamplerAdapter(torch.utils.data.Sampler):
    """torch ``Sampler`` shell around the torch-free :class:`MutableSubsetSampler`."""

    def __init__(self, inner: MutableSubsetSampler) -> None:
        self._inner = inner

    def __iter__(self):  # noqa: ANN201
        return iter(self._inner)

    def __len__(self) -> int:
        return len(self._inner)


class _IndexedDataset:
    """Adds ``__row_index__`` to each item so hooks can track canonical row ids."""

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, i: int) -> Any:  # noqa: ANN401
        item = self._dataset[i]
        if isinstance(item, dict):
            item = dict(item)
            item["__row_index__"] = int(i)
        return item

    def __getitems__(self, indices: list[int]) -> list[Any]:
        # torch's fetcher prefers ``__getitems__`` when present (and ``datasets.Dataset``
        # has one) — without this override the batch path would bypass ``__getitem__``.
        items = self._dataset.__getitems__(indices)
        out = []
        for item, i in zip(items, indices, strict=False):
            if isinstance(item, dict):
                item = dict(item)
                item["__row_index__"] = int(i)
            out.append(item)
        return out

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._dataset, name)


class _IndexedCollator:
    """Pops ``__row_index__`` before the real collator and re-attaches it to the batch."""

    def __init__(self, collator: Any) -> None:
        self._collator = collator

    def __call__(self, features: list[Any]) -> Any:  # noqa: ANN401
        indices = [f.pop("__row_index__", None) if isinstance(f, dict) else None for f in features]
        batch = self._collator(features)
        if any(i is not None for i in indices):
            batch["__row_index__"] = [int(i) if i is not None else -1 for i in indices]
        return batch

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._collator, name)


def make_online_trainer(base_cls: type) -> type:
    """Build a trainer subclass of ``base_cls`` (LF's SFT trainer) wired with online hooks."""

    class OnlineOpsTrainer(base_cls):  # type: ignore[valid-type, misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._reweight_hooks = [
                h for h in _ACTIVE_HOOKS if getattr(h, "hook_point", "") == "reweight"
            ]
            self._select_hooks = [
                h for h in _ACTIVE_HOOKS if getattr(h, "hook_point", "") == "select"
            ]
            self._select_sampler_ref: dict[str, Any] = {"sampler": None}
            step_hooks = [h for h in _ACTIVE_HOOKS if getattr(h, "hook_point", "") == "step"]
            if step_hooks:
                self.add_callback(StepHookCallback(step_hooks))
            if self._select_hooks:
                self.add_callback(SelectHookCallback(self._select_hooks, self._select_sampler_ref))
            if self._reweight_hooks or self._select_hooks:
                self.train_dataset = _IndexedDataset(self.train_dataset)
                self.data_collator = _IndexedCollator(self.data_collator)

        def _get_train_sampler(self, *args: Any, **kwargs: Any):  # noqa: ANN202
            base_sampler = super()._get_train_sampler(*args, **kwargs)
            if not self._select_hooks or base_sampler is None:
                return base_sampler
            inner = MutableSubsetSampler(base_sampler, len(self.train_dataset))
            self._select_sampler_ref["sampler"] = inner
            self._select_sampler_ref["n_rows"] = len(self.train_dataset)
            return _SamplerAdapter(inner)

        def _get_collator_with_removed_columns(self, data_collator, description=None):  # noqa: ANN001, ANN202
            # HF wraps the OUTERMOST collator here (RemoveColumnsCollator strips every key not
            # in the model signature); whitelist our row-index key so it survives to the
            # indexed collator underneath.
            collator = super()._get_collator_with_removed_columns(
                data_collator, description=description
            )
            keep = getattr(collator, "signature_columns", None)
            if isinstance(keep, list) and "__row_index__" not in keep:
                keep.append("__row_index__")
            return collator

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # noqa: ANN001
            row_indices = inputs.pop("__row_index__", None)
            needs_losses = (self._reweight_hooks or self._select_hooks) and "labels" in inputs
            if not needs_losses:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
            per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            ).view(shift_labels.size(0), -1)
            mask = (shift_labels != -100).to(per_token.dtype)
            per_sample = (per_token * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            batch = {
                "losses": per_sample.detach().float().cpu().tolist(),
                "row_indices": row_indices,
                "domains": None,  # reserved by the protocol
                "step": getattr(self.state, "global_step", 0) or 0,
                "epoch": int(getattr(self.state, "epoch", None) or 0),
            }
            for hook in self._select_hooks:
                observe = getattr(hook, "observe", None)
                if observe is not None:
                    with suppress(Exception):
                        observe(batch)
            factors = apply_reweight(batch, self._reweight_hooks)
            weights = torch.tensor(factors, device=per_sample.device, dtype=per_sample.dtype)
            loss = (per_sample * weights).mean()
            return (loss, outputs) if return_outputs else loss

    return OnlineOpsTrainer
