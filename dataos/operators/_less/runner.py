from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from dataos.core.sandbox import Sandbox
from dataos.operators._less.config import LessConfig

STAGE_NAMES = ("warmup", "train_grads", "val_grads", "select")


@dataclass(slots=True)
class StageResult:
    name: str
    cache_status: str
    cache_key: str | None
    cache_path: str | None
    log_path: str | None


@dataclass(slots=True)
class PipelineResult:
    selected_indices: list[int]
    stages: list[StageResult]
    cache_root: Path | None


def run_stub_pipeline(
    config: LessConfig,
    pool_dataset: Dataset,
    sandbox: Sandbox,
) -> PipelineResult:
    """Bypass cache + LESS subprocess; deterministic shuffle-by-seed selection.

    Used for local CI and any test that should not depend on a working LESS
    repo or GPU. Each LESS stage produces a tiny placeholder log so the
    artifact / sandbox shape mirrors the real pipeline.
    """
    rng = random.Random(config.seed)
    indices = list(range(pool_dataset.num_rows))
    rng.shuffle(indices)
    keep = _resolve_keep_count(config, pool_dataset.num_rows)
    selected = sorted(indices[:keep])

    stages: list[StageResult] = []
    for stage_name in STAGE_NAMES:
        log_path = sandbox.log(
            f"less_{stage_name}",
            (
                f"stage={stage_name}\n"
                f"mode=stub\n"
                f"pool_rows={pool_dataset.num_rows}\n"
                f"selected={len(selected)}\n"
            ),
        )
        stages.append(
            StageResult(
                name=stage_name,
                cache_status="skipped",
                cache_key=None,
                cache_path=None,
                log_path=str(log_path),
            )
        )
    return PipelineResult(selected_indices=selected, stages=stages, cache_root=None)


def run_less_pipeline(
    config: LessConfig,
    pool_artifact_id: str,
    pool_dataset: Dataset,
    target_artifact_id: str,
    target_dataset: Dataset,
    sandbox: Sandbox,
) -> PipelineResult:
    """Real LESS pipeline orchestration via subprocess (lands in commit 2)."""
    raise NotImplementedError(
        "less: mode='less' real pipeline lands in commit 2; use mode='stub' for CI."
    )


def _resolve_keep_count(config: LessConfig, pool_size: int) -> int:
    if config.top_k is not None:
        if config.top_k > pool_size:
            raise ValueError(f"less: top_k={config.top_k} exceeds pool size {pool_size}")
        return config.top_k
    fraction = config.top_fraction if config.top_fraction is not None else 0.0
    keep = max(1, int(round(pool_size * fraction)))
    return min(keep, pool_size)
