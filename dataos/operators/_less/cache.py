from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from dataos.operators._less.config import LessConfig

CACHE_VERSION = "1"
ENV_CACHE_DIR = "DATAOS_LESS_CACHE"


def resolve_cache_root(sandbox_root: Path, config: LessConfig) -> Path:
    """Pick the workspace-level cache root for LESS intermediates.

    Resolution order:
    1. ``DATAOS_LESS_CACHE`` env var (deployment override).
    2. ``config.cache_dir`` kwarg (per-call override).
    3. Inferred ``<workspace>/cache/less`` via ``sandbox_root.parent.parent``.
       Inference accepted **only** when the candidate looks like a real
       DataOS workspace — either it already contains a ``sandboxes/``
       subdirectory (matches ``run_task`` layout) or its directory name is
       literally ``workspace``. This stops unit tests with
       ``Sandbox(tmp_path / "sandbox")`` from polluting ``tmp_path.parent``.
    4. ``~/.cache/dataos/less`` fallback.
    """
    env_value = os.environ.get(ENV_CACHE_DIR)
    if env_value:
        return Path(env_value).expanduser()
    if config.cache_dir:
        return Path(config.cache_dir).expanduser()
    inferred_workspace = sandbox_root.parent.parent
    looks_like_workspace = inferred_workspace.exists() and (
        (inferred_workspace / "sandboxes").is_dir()
        or inferred_workspace.name == "workspace"
    )
    if looks_like_workspace and inferred_workspace != sandbox_root:
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

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Hold an exclusive ``fcntl.flock`` on a sibling lock file.

        Serialises concurrent workers contending for the same cache key. The
        intended pattern is:

            with entry.lock():
                if entry.hit():        # double-check after acquiring lock
                    skip subprocess
                else:
                    run subprocess
                    entry.write_meta(...)

        ``fcntl.flock`` locks per-process; tests use ``multiprocessing`` to
        exercise contention. NFS / SMB filesystems can degrade flock semantics
        — DataOS does not deploy on those, so we accept the limitation.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        lock_path = self.path / ".lock"
        with lock_path.open("w") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def make_cache_entry(cache_root: Path, stage: str, key: str) -> CacheEntry:
    return CacheEntry(stage=stage, key=key, path=cache_root / stage / key)
