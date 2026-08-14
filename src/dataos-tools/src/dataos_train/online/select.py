"""Torch-free mutable subset sampler backing the ``select`` hook point.

The ``select`` protocol (see :mod:`dataos_train.online`) lets an operator decide — at an
epoch boundary — which dataset rows the next epoch trains on. The backend wraps the
trainer's train sampler with :class:`MutableSubsetSampler`; a hook decision simply swaps
the allowed-index set, and the new subset takes effect from the next epoch because the
dataloader re-iterates the sampler every epoch.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator


class MutableSubsetSampler:
    """Sampler wrapper whose yielded indices can be narrowed between epochs.

    ``allowed=None`` means pass-through (full dataset). ``set_allowed(indices)`` restricts
    subsequent iterations to those row indices; ``set_allowed(None)`` restores pass-through.
    """

    def __init__(self, base: Iterable[int], n: int) -> None:
        self._base = base
        self._n = int(n)
        self._allowed: frozenset[int] | None = None

    def set_allowed(self, indices: Iterable[int] | None) -> None:
        self._allowed = None if indices is None else frozenset(int(i) for i in indices)

    @property
    def allowed(self) -> frozenset[int] | None:
        return self._allowed

    def __iter__(self) -> Iterator[int]:
        if self._allowed is None:
            yield from self._base
            return
        allowed = self._allowed
        for i in self._base:
            if i in allowed:
                yield i

    def __len__(self) -> int:
        return self._n if self._allowed is None else len(self._allowed)
