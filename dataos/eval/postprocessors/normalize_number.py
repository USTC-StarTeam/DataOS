from __future__ import annotations


def normalize_number(value: str) -> str:
    cleaned = value.replace("$", "").replace(",", "").strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return cleaned
