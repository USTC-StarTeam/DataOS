from __future__ import annotations


def exact_match(prediction: str | None, reference: str | None) -> float:
    return 1.0 if prediction is not None and prediction == reference else 0.0


def aggregate_exact_match(sample_scores: list[float]) -> float:
    if not sample_scores:
        return 0.0
    return sum(sample_scores) / len(sample_scores)
