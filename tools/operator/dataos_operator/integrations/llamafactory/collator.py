"""LLaMA-Factory collator wrapper that preserves DataOS runtime metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataOSLanguageModelingCollator:
    """Wrap a LLaMA-Factory collator and keep DataOS ids/domains in the batch."""

    base_collator: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        from llamafactory.data.collator import _pop_dataos_runtime_fields

        dataos_runtime_fields = _pop_dataos_runtime_fields(features)
        batch = self.base_collator(features)
        batch.update(dataos_runtime_fields)
        return batch
