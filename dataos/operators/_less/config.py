from __future__ import annotations

from dataclasses import dataclass
from typing import Any

VALID_MODES = ("less", "stub")
REQUIRED_KWARGS = ("target_artifact_id", "base_model")
RUNTIME_INJECTED_KWARGS = {"task", "experiment"}
ALLOWED_KWARGS = {
    "target_artifact_id",
    "base_model",
    "mode",
    "top_k",
    "top_fraction",
    "seed",
    "warmup_epochs",
    "projection_dim",
    "lora_r",
    "lora_alpha",
    "learning_rate",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "less_repo_path",
    "cache_dir",
    "force_recompute",
    "python_executable",
}


@dataclass(slots=True)
class LessConfig:
    target_artifact_id: str
    base_model: str
    mode: str
    top_k: int | None
    top_fraction: float | None
    seed: int
    warmup_epochs: int
    projection_dim: int
    lora_r: int
    lora_alpha: int
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    less_repo_path: str | None
    cache_dir: str | None
    force_recompute: bool
    python_executable: str | None

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> "LessConfig":
        params = {key: value for key, value in kwargs.items() if key not in RUNTIME_INJECTED_KWARGS}
        unknown = set(params) - ALLOWED_KWARGS
        if unknown:
            keys = ", ".join(sorted(unknown))
            raise ValueError(f"less: unsupported params: {keys}")
        for key in REQUIRED_KWARGS:
            if not params.get(key):
                raise ValueError(f"less: '{key}' is required")
        mode = str(params.get("mode", "less"))
        if mode not in VALID_MODES:
            raise ValueError(f"less: 'mode' must be one of {list(VALID_MODES)}")

        top_k = params.get("top_k")
        top_fraction = params.get("top_fraction")
        if top_k is None and top_fraction is None:
            raise ValueError("less: either 'top_k' or 'top_fraction' is required")
        if top_k is not None and top_fraction is not None:
            raise ValueError("less: 'top_k' and 'top_fraction' are mutually exclusive")
        if top_k is not None and int(top_k) <= 0:
            raise ValueError("less: 'top_k' must be > 0")
        if top_fraction is not None and not (0 < float(top_fraction) <= 1):
            raise ValueError("less: 'top_fraction' must be in (0, 1]")

        if mode == "less" and not params.get("less_repo_path"):
            raise ValueError("less: 'less_repo_path' is required when mode='less'")

        return cls(
            target_artifact_id=str(params["target_artifact_id"]),
            base_model=str(params["base_model"]),
            mode=mode,
            top_k=int(top_k) if top_k is not None else None,
            top_fraction=float(top_fraction) if top_fraction is not None else None,
            seed=int(params.get("seed", 42)),
            warmup_epochs=int(params.get("warmup_epochs", 4)),
            projection_dim=int(params.get("projection_dim", 8192)),
            lora_r=int(params.get("lora_r", 128)),
            lora_alpha=int(params.get("lora_alpha", 512)),
            learning_rate=float(params.get("learning_rate", 2e-5)),
            per_device_train_batch_size=int(params.get("per_device_train_batch_size", 1)),
            gradient_accumulation_steps=int(params.get("gradient_accumulation_steps", 32)),
            less_repo_path=str(params["less_repo_path"]) if params.get("less_repo_path") else None,
            cache_dir=str(params["cache_dir"]) if params.get("cache_dir") else None,
            force_recompute=bool(params.get("force_recompute", False)),
            python_executable=(
                str(params["python_executable"])
                if params.get("python_executable")
                else None
            ),
        )
