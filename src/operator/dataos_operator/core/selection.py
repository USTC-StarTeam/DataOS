"""Small deterministic helpers shared by selection operators."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


def stratum_label(values: Sequence[Any]) -> str:
    return "__all__" if not values else "|".join(str(value) for value in values)


def stable_fraction(seed: int, value: Any) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 1)


def numeric(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Field {key!r} must be numeric, got boolean")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Field {key!r} is not numeric: {value!r}") from exc
    raise ValueError(f"Field {key!r} is not numeric: {value!r}")


def allocate_quotas(
    capacities: Mapping[str, int],
    target_size: int,
    *,
    weights: Mapping[str, float] | None = None,
    minimum_per_stratum: int = 0,
) -> dict[str, int]:
    """Allocate an exact capacity-aware weighted quota with deterministic remainders."""

    if target_size < 0:
        raise ValueError("target_size must be non-negative")
    if target_size > sum(capacities.values()):
        raise ValueError("target_size exceeds total stratum capacity")
    if minimum_per_stratum < 0:
        raise ValueError("minimum_per_stratum must be non-negative")

    quotas = {label: min(capacity, minimum_per_stratum) for label, capacity in capacities.items()}
    if sum(quotas.values()) > target_size:
        raise ValueError("minimum_per_stratum exceeds target_size")

    while sum(quotas.values()) < target_size:
        remaining = target_size - sum(quotas.values())
        open_labels = [label for label, capacity in capacities.items() if quotas[label] < capacity]
        masses = {
            label: (capacities[label] - quotas[label]) * float((weights or {}).get(label, 1.0))
            for label in open_labels
        }
        if any(mass < 0 for mass in masses.values()):
            raise ValueError("stratum weights must be non-negative")
        total_mass = sum(masses.values())
        if not open_labels or total_mass <= 0:
            raise ValueError("positive-capacity strata need positive allocation weight")

        raw = {label: remaining * masses[label] / total_mass for label in open_labels}
        additions = {
            label: min(capacities[label] - quotas[label], int(raw[label])) for label in open_labels
        }
        allocated = sum(additions.values())
        for label, addition in additions.items():
            quotas[label] += addition
        if allocated:
            continue

        label = max(
            open_labels,
            key=lambda item: (raw[item] - int(raw[item]), masses[item], item),
        )
        quotas[label] += 1
    return quotas
