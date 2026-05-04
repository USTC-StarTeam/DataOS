from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataos.operators._less.config import LessConfig

CACHE_VERSION = "1"
ENV_CACHE_DIR = "DATAOS_LESS_CACHE"


def resolve_cache_root(sandbox_root: Path, config: LessConfig) -> Path:
    """Pick the workspace-level cache root for LESS intermediates.

    Resolution order:
    1. ``DATAOS_LESS_CACHE`` env var.
    2. ``config.cache_dir`` kwarg.
    3. ``<workspace>/cache/less`` derived from ``sandbox_root.parent.parent``
       (matches the runner.py layout: ``<workspace>/sandboxes/<exp_id>``).
    4. ``~/.cache/dataos/less`` fallback.
    """
    env_value = os.environ.get(ENV_CACHE_DIR)
    if env_value:
        return Path(env_value).expanduser()
    if config.cache_dir:
        return Path(config.cache_dir).expanduser()
    inferred_workspace = sandbox_root.parent.parent
    if inferred_workspace.exists() and inferred_workspace != sandbox_root:
        return inferred_workspace / "cache" / "less"
    return Path("~/.cache/dataos/less").expanduser()


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def warmup_key(pool_artifact_id: str, config: LessConfig) -> str:
    return _stable_hash(
        {
            "version": CACHE_VERSION,
            "stage": "warmup",
            "pool_artifact_id": pool_artifact_id,
            "base_model": config.base_model,
            "warmup_epochs": config.warmup_epochs,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "learning_rate": config.learning_rate,
            "per_device_train_batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "seed": config.seed,
        }
    )


def train_grads_key(warmup_key_value: str, config: LessConfig) -> str:
    return _stable_hash(
        {
            "version": CACHE_VERSION,
            "stage": "train_grads",
            "warmup_key": warmup_key_value,
            "projection_dim": config.projection_dim,
        }
    )


def val_grads_key(warmup_key_value: str, target_artifact_id: str, config: LessConfig) -> str:
    return _stable_hash(
        {
            "version": CACHE_VERSION,
            "stage": "val_grads",
            "warmup_key": warmup_key_value,
            "target_artifact_id": target_artifact_id,
            "projection_dim": config.projection_dim,
        }
    )


@dataclass(slots=True)
class CacheEntry:
    stage: str
    key: str
    path: Path

    def hit(self) -> bool:
        return (self.path / "cache_meta.json").exists()

    def reserve(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path

    def write_meta(self, payload: dict[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        record = {"stage": self.stage, "key": self.key, **payload}
        (self.path / "cache_meta.json").write_text(
            json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    def read_meta(self) -> dict[str, Any]:
        return json.loads((self.path / "cache_meta.json").read_text(encoding="utf-8"))


def make_cache_entry(cache_root: Path, stage: str, key: str) -> CacheEntry:
    return CacheEntry(stage=stage, key=key, path=cache_root / stage / key)
