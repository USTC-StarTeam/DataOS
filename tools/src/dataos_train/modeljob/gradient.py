"""Pure numpy helpers for LESS-style gradient-feature scoring (arXiv 2402.04333).

Everything here is torch-free and deterministic given ``seed`` so the math is
unit-testable without the training stack and reproducible across records. ALL candidate
and target records MUST be projected with the SAME seed — that yields one shared
Rademacher projection matrix; mixing per-record seeds makes cosine similarities
incomparable (a historical bug this module is designed to prevent).

The model-side glue (LoRA attachment, forward/backward) lives in
:mod:`dataos_train.modeljob.score_entry` and lazily imports torch/peft at job time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

# Rows per streamed Rademacher block. Fixed so the draw pattern — and therefore the
# projection — depends only on ``seed``, never on the caller's chunking choice.
_CHUNK_ROWS = 1 << 16

SIGN_EPS = 1e-12  # keeps sign_normalize finite at g == 0 without perturbing signs


def rademacher_matrix(n_rows: int, dim: int, seed: int) -> NDArray[np.float64]:
    """The ``(n_rows, dim)`` Rademacher projection block Π with entries in {±1}/√dim.

    Drawn from ``np.random.default_rng(seed)`` consumed in fixed ``_CHUNK_ROWS`` row
    blocks — the same consumption pattern :func:`project` uses — so
    ``project(v, dim, seed) == v @ rademacher_matrix(len(v), dim, seed)``.
    """
    if n_rows < 0 or dim < 1:
        raise ValueError(f"need n_rows >= 0 and dim >= 1, got {n_rows=} {dim=}")
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(dim)
    blocks = [
        rng.choice((-1.0, 1.0), size=(rows, dim)) * scale
        for start in range(0, n_rows, _CHUNK_ROWS)
        for rows in [min(_CHUNK_ROWS, n_rows - start)]
    ]
    if not blocks:
        return np.zeros((0, dim))
    return np.concatenate(blocks) if len(blocks) > 1 else blocks[0]


def project(
    vec: Sequence[float] | NDArray[np.floating], dim: int, seed: int
) -> NDArray[np.float64]:
    """Project a (possibly huge) gradient vector to ``dim`` dims, sparse JL-style.

    For each coordinate i a deterministic bucket j(i) ∈ [0, dim) and sign s(i) ∈ {±1}
    are drawn (fixed chunk size, so the draw pattern depends only on ``seed``), and
    out[j] += s·v[i]/√dim is accumulated. This is a JL-quality projection like the
    dense Rademacher one, but O(n) instead of O(n·dim) — at gradient scale (≥10⁷
    dims) the dense matmul is minutes per record, this is milliseconds.
    """
    if dim < 1:
        raise ValueError(f"projection dim must be >= 1, got {dim}")
    v = np.asarray(vec, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(dim)
    out = np.zeros(dim)
    for start in range(0, v.size, _CHUNK_ROWS):
        block = v[start : start + _CHUNK_ROWS]
        idx = rng.integers(0, dim, size=block.size)
        signs = rng.choice((-1.0, 1.0), size=block.size)
        np.add.at(out, idx, signs * block * scale)
    return out


def sign_normalize(
    grad: Sequence[float] | NDArray[np.floating], eps: float = SIGN_EPS
) -> NDArray[np.float64]:
    """LESS Adam-state approximation: ``g / (|g| + eps)`` element-wise (sign gradient).

    The paper normalizes by the Adam second moment (``m / √(v + ε)``); at scoring time
    no optimizer state exists, so v1 uses this sign-grad surrogate — shown usable in the
    paper's own ablations. Deliberate approximation; v2 may swap in a real Adam state.
    """
    g = np.asarray(grad, dtype=np.float64)
    return g / (np.abs(g) + eps)


def flatten_grads(grads: Sequence[NDArray[np.floating]]) -> NDArray[np.float64]:
    """Concatenate per-parameter gradient arrays into one flat vector.

    Order follows the caller's parameter order (``model.parameters()`` registration
    order), which is fixed for a given model+adapter setup, so features line up across
    records.
    """
    flats = [np.asarray(g, dtype=np.float64).ravel() for g in grads]
    return np.concatenate(flats) if flats else np.zeros(0)


def cosine(
    a: Sequence[float] | NDArray[np.floating], b: Sequence[float] | NDArray[np.floating]
) -> float:
    """Cosine similarity; 0.0 when either vector is zero (no direction to compare)."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y) / denom if denom > 0 else 0.0


def l2_norm(vec: Sequence[float] | NDArray[np.floating]) -> float:
    """L2 norm of a (projected) feature — the no-target fallback score."""
    return float(np.linalg.norm(np.asarray(vec, dtype=np.float64)))


def mean_direction(vectors: Sequence[NDArray[np.floating]]) -> NDArray[np.float64]:
    """Mean of the given features, L2-normalized — the target direction for cosine."""
    arrs = [np.asarray(v, dtype=np.float64).ravel() for v in vectors]
    if not arrs:
        return np.zeros(0)
    mean = np.mean(np.stack(arrs), axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 0 else mean
