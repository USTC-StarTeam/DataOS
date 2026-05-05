"""LESS (Low-rank gradiEnt Similarity Search) data-selection operator.

Wraps the official ``princeton-nlp/LESS`` pipeline as a single deterministic
DataOS operator. Internally it runs four LESS stages — warmup LoRA training,
training-data gradient projection, validation-data gradient projection, and
top-k influence selection — but exposes a flat operator contract to the plan
and runtime layers. Heavy intermediates are content-addressed and cached at
the workspace level so reuse across target tasks is automatic when the pool
and warmup hyperparameters are unchanged.

Pool / target identification rule
---------------------------------

The operator distinguishes the two ``DatasetArtifact`` inputs as follows:

- **target** = the artifact in ``inputs`` whose ``artifact_id`` matches the
  required ``target_artifact_id`` kwarg exactly. Required and unambiguous.
- **pool** = the **most recently added** ``DatasetArtifact`` in ``inputs``
  whose id is **not** the target. ``find_pool_dataset`` scans
  ``reversed(inputs)`` and returns the first match, so input ordering matters
  only when multiple non-target datasets are present — the latest wins.

When ``inputs`` contains a single ``DatasetArtifact`` and that artifact is
the target, no pool is available and the operator raises. When two or more
non-target datasets are present, only the latest is consumed; today the
operator does not accept an explicit ``pool_artifact_id`` kwarg, so callers
that need a non-default pool must reorder ``inputs`` so the desired pool
comes last. (Explicit ``pool_artifact_id`` is a candidate future kwarg —
do not assume it exists yet.)

Spec fields (kwargs)
--------------------

- ``target_artifact_id`` (required): artifact_id of the target-task
  ``DatasetArtifact`` already present in ``inputs``.
- ``base_model`` (required): warmup base model path or HF id.
- ``top_k`` or ``top_fraction`` (one is required): selection size.
- ``mode`` (default ``"less"``): ``"less"`` invokes the official pipeline;
  ``"stub"`` is a fast deterministic shuffle-by-seed for local CI.
- LESS hyperparams (optional): ``warmup_epochs`` (4), ``projection_dim``
  (8192), ``lora_r`` (128), ``lora_alpha`` (512), ``learning_rate`` (2e-5),
  ``per_device_train_batch_size`` (1), ``gradient_accumulation_steps`` (32),
  ``seed`` (42).
- ``less_repo_path`` (required when ``mode='less'``): local clone of
  ``princeton-nlp/LESS``.
- ``cache_dir`` (optional): override for the cross-experiment cache root.
  Resolution order: ``DATAOS_LESS_CACHE`` env > this kwarg > inferred
  ``<workspace>/cache/less`` > ``~/.cache/dataos/less`` fallback.
- ``python_executable`` (optional): override for the python interpreter that
  runs ``-m less.<module>`` subprocesses. Resolution order:
  this kwarg > ``DATAOS_LESS_PYTHON`` env > ``sys.executable``.
- ``force_recompute`` (default ``False``): bypass cache hits.
"""

from __future__ import annotations

import json

from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox
from dataos.operators._less.config import LessConfig
from dataos.operators._less.io import (
    build_selected_dataset,
    check_required_columns,
    find_pool_dataset,
    find_target_dataset,
    select_train_split,
)
from dataos.operators._less.runner import (
    PipelineResult,
    run_less_pipeline,
    run_stub_pipeline,
)


class LessSelectOperator(Operator):
    name = "less"
    description = (
        "Select an influential training subset for a target task via LESS "
        "(low-rank gradient similarity search)."
    )

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        config = LessConfig.from_kwargs(kwargs)
        target = find_target_dataset(inputs, config.target_artifact_id)
        pool = find_pool_dataset(inputs, config.target_artifact_id)

        pool_obj = pool.load()
        target_obj = target.load()
        check_required_columns(pool_obj, "pool")
        check_required_columns(target_obj, "target")

        pool_train, pool_split = select_train_split(pool_obj)
        target_train, _ = select_train_split(target_obj)

        result = self._dispatch(config, pool, pool_train, target, target_train, sandbox)

        selected_obj = build_selected_dataset(pool_obj, pool_split, result.selected_indices)

        artifact_id = make_artifact_id("dataset")
        out = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{pool.name}-less",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[pool.artifact_id, target.artifact_id],
            producer=self.name,
            meta=_build_meta(config, pool_train.num_rows, result),
        )
        out.save(selected_obj)

        manifest = _build_manifest(config, pool, target, pool_train.num_rows, result)
        (out.path / "selection_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

        sandbox.log(
            self.name,
            (
                f"pool={pool.artifact_id}\n"
                f"target={target.artifact_id}\n"
                f"output={out.artifact_id}\n"
                f"mode={config.mode}\n"
                f"pool_count={pool_train.num_rows}\n"
                f"selected={len(result.selected_indices)}\n"
            ),
        )
        return out

    def _dispatch(
        self,
        config: LessConfig,
        pool: DatasetArtifact,
        pool_train,
        target: DatasetArtifact,
        target_train,
        sandbox: Sandbox,
    ) -> PipelineResult:
        if config.mode == "stub":
            return run_stub_pipeline(config, pool_train, sandbox)
        return run_less_pipeline(
            config=config,
            pool_artifact_id=pool.artifact_id,
            pool_dataset=pool_train,
            target_artifact_id=target.artifact_id,
            target_dataset=target_train,
            sandbox=sandbox,
        )


def _stage_record(stage) -> dict:
    return {
        "name": stage.name,
        "cache_status": stage.cache_status,
        "cache_key": stage.cache_key,
        "cache_path": stage.cache_path,
        "log_path": stage.log_path,
    }


def _build_meta(config: LessConfig, pool_count: int, result: PipelineResult) -> dict:
    return {
        "less_mode": config.mode,
        "less_base_model": config.base_model,
        "less_top_k": config.top_k,
        "less_top_fraction": config.top_fraction,
        "less_pool_count": pool_count,
        "less_selected_count": len(result.selected_indices),
        "less_target_artifact_id": config.target_artifact_id,
        "less_warmup_epochs": config.warmup_epochs,
        "less_projection_dim": config.projection_dim,
        "less_seed": config.seed,
        "less_cache_root": str(result.cache_root) if result.cache_root is not None else None,
        "less_stages": [_stage_record(stage) for stage in result.stages],
    }


def _build_manifest(
    config: LessConfig,
    pool: DatasetArtifact,
    target: DatasetArtifact,
    pool_count: int,
    result: PipelineResult,
) -> dict:
    return {
        "less_mode": config.mode,
        "base_model": config.base_model,
        "top_k": config.top_k,
        "top_fraction": config.top_fraction,
        "pool_artifact_id": pool.artifact_id,
        "target_artifact_id": target.artifact_id,
        "pool_count": pool_count,
        "selected_count": len(result.selected_indices),
        "warmup_epochs": config.warmup_epochs,
        "projection_dim": config.projection_dim,
        "seed": config.seed,
        "cache_root": str(result.cache_root) if result.cache_root is not None else None,
        "stages": [_stage_record(stage) for stage in result.stages],
        "selected_indices": result.selected_indices,
    }
