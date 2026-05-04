"""LESS subprocess command construction.

Centralises every external CLI call so that signature drift in the upstream
``princeton-nlp/LESS`` repo can be patched in one place. The exact arg names
mirror the LESS module entrypoints as of early 2026 and remain best-effort
until verified on real hardware (see commit-3 hardware run on bdaa_5090-1).

If LESS's CLI changes, edit only this file. The orchestration in
``runner.py`` does not depend on argument specifics — only on the stage
producing the documented output files.
"""

from __future__ import annotations

from pathlib import Path

from dataos.operators._less.config import LessConfig

PYTHON = "python"


def warmup_command(
    config: LessConfig,
    pool_jsonl: Path,
    output_dir: Path,
) -> list[str]:
    """Stage 1 — warmup LoRA training. Output: ``<output_dir>/checkpoint-*/``."""
    return [
        PYTHON,
        "-m",
        "less.train.train",
        "--model_name_or_path",
        config.base_model,
        "--train_files",
        str(pool_jsonl),
        "--output_dir",
        str(output_dir),
        "--num_train_epochs",
        str(config.warmup_epochs),
        "--learning_rate",
        str(config.learning_rate),
        "--per_device_train_batch_size",
        str(config.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(config.gradient_accumulation_steps),
        "--lora_r",
        str(config.lora_r),
        "--lora_alpha",
        str(config.lora_alpha),
        "--seed",
        str(config.seed),
        "--save_strategy",
        "epoch",
        "--logging_steps",
        "1",
    ]


def grad_command(
    config: LessConfig,
    *,
    gradient_type: str,
    train_files: Path,
    model_path: Path,
    output_path: Path,
) -> list[str]:
    """Stage 2/3 — projected gradients. ``gradient_type`` is ``"sgd"`` (train pool)
    or ``"adam"`` (target task). Output: arbitrary files under ``output_path``.
    """
    if gradient_type not in {"sgd", "adam"}:
        raise ValueError(f"less: gradient_type must be 'sgd' or 'adam' (got {gradient_type!r})")
    return [
        PYTHON,
        "-m",
        "less.data_selection.get_info",
        "--info_type",
        "grads",
        "--gradient_type",
        gradient_type,
        "--train_files",
        str(train_files),
        "--model_path",
        str(model_path),
        "--output_path",
        str(output_path),
        "--gradient_projection_dimension",
        str(config.projection_dim),
    ]


def matching_command(
    train_grads_dir: Path,
    val_grads_dir: Path,
    output_path: Path,
) -> list[str]:
    """Stage 4a — cosine similarity matching. Output: scores at ``output_path``."""
    return [
        PYTHON,
        "-m",
        "less.data_selection.matching",
        "--gradient_path",
        str(train_grads_dir),
        "--validation_gradient_path",
        str(val_grads_dir),
        "--output_path",
        str(output_path),
    ]


# Inline ranking script. Loads the matching output (torch tensor on real
# hardware; JSON list as a test-friendly fallback) and writes a sorted
# top-k indices JSON file. Done in a subprocess so monkeypatching covers it.
RANK_SCRIPT = """
import json
import sys

scores_path = sys.argv[1]
keep = int(sys.argv[2])
output_path = sys.argv[3]

try:
    import torch
    raw = torch.load(scores_path)
    if isinstance(raw, dict):
        raw = next(iter(raw.values()))
    flat = raw.flatten().tolist()
except Exception:
    with open(scores_path, "r", encoding="utf-8") as f:
        flat = json.load(f)

ranked = sorted(range(len(flat)), key=lambda i: -flat[i])[:keep]
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sorted(ranked), f)
"""


def rank_command(scores_path: Path, keep: int, indices_path: Path) -> list[str]:
    return [PYTHON, "-c", RANK_SCRIPT, str(scores_path), str(keep), str(indices_path)]
