"""LESS subprocess command construction.

Centralises every external CLI call so that signature drift in the upstream
``princeton-nlp/LESS`` repo can be patched in one place. The orchestration
in ``runner.py`` does not depend on argument specifics — only on the stage
producing the documented output files.

CLI shapes here mirror princeton-nlp/LESS @ commit ``8abf962`` (May 2026
verification on bdaa_5090-1) and the conventions in
``less/scripts/{train,get_info,data_selection}/*.sh``:

  - warmup: ``torchrun -m less.train.train`` (not ``python -m``); needs the
    full HF-Trainer flag surface plus LESS-specific LoRA flags.
  - gradients: ``python3 -m less.data_selection.get_info`` with
    ``--train_file`` (singular) and ``--gradient_type`` ∈ {sgd, adam}.
    Convention: train pool uses ``adam``, target task uses ``sgd``.
  - select: LESS's matching script is hardcoded for {mmlu, bbh, tydiqa}
    (see ``N_SUBTASKS`` in ``less.data_selection.matching``); for arbitrary
    target tasks we replace it with an inline cosine-similarity top-k
    computed in a ``python3 -c`` subprocess so monkeypatching still works.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dataos.operators._less.config import LessConfig

ENV_PYTHON = "DATAOS_LESS_PYTHON"
TORCHRUN = "torchrun"


def python_for(config: LessConfig) -> str:
    """Resolve which python executable to use for ``-m less.<module>`` calls.

    Precedence: ``config.python_executable`` > ``$DATAOS_LESS_PYTHON`` env var
    > ``sys.executable``. The latter ensures we drive the same interpreter that
    imported DataOS itself, which is correct for ``uv run`` / venv setups
    where the LESS deps were installed.
    """
    if config.python_executable:
        return config.python_executable
    env_value = os.environ.get(ENV_PYTHON)
    if env_value:
        return env_value
    return sys.executable


def warmup_command(
    config: LessConfig,
    pool_jsonl: Path,
    output_dir: Path,
) -> list[str]:
    """Stage 1 — warmup LoRA training. Output: ``<output_dir>/checkpoint-*/``.

    The ``--master-port`` is offset from torchrun's default 29500 by
    ``config.seed`` so that concurrent operator invocations on the same host
    (Phase 3 alignment runs across 3 GPUs) do not collide on the rendezvous
    socket. The offset is taken modulo a 16k-port window starting at 29500
    to stay well clear of low-numbered well-known ports.
    """
    rdzv_port = 29500 + (config.seed % 16384)
    return [
        TORCHRUN,
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master-port",
        str(rdzv_port),
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
        "--lora",
        "True",
        "--lora_r",
        str(config.lora_r),
        "--lora_alpha",
        str(config.lora_alpha),
        "--lora_dropout",
        "0.1",
        "--lora_target_modules",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "--seed",
        str(config.seed),
        "--data_seed",
        str(config.seed),
        "--save_strategy",
        "epoch",
        "--logging_steps",
        "1",
        "--bf16",
        "True",
        "--tf32",
        "False",
        "--fp16",
        "False",
        "--do_train",
        "True",
        "--report_to",
        "none",
        "--optim",
        "adamw_torch",
        "--lr_scheduler_type",
        "linear",
        "--warmup_ratio",
        "0.03",
        "--weight_decay",
        "0.0",
        "--max_seq_length",
        "2048",
        "--use_fast_tokenizer",
        "True",
        "--overwrite_output_dir",
        "True",
        "--percentage",
        "1.0",
    ]


def grad_command(
    config: LessConfig,
    *,
    gradient_type: str,
    train_files: Path,
    model_path: Path,
    output_path: Path,
) -> list[str]:
    """Stage 2/3 — projected gradients via ``less.data_selection.get_info``.

    ``gradient_type`` is ``"adam"`` for the training pool (matches the warmup
    optimizer) and ``"sgd"`` for the target task. ``train_files`` is a single
    JSONL path despite the historical plural name in DataOS code; LESS's CLI
    flag is ``--train_file`` (singular) and accepts exactly one path.
    """
    if gradient_type not in {"sgd", "adam"}:
        raise ValueError(f"less: gradient_type must be 'sgd' or 'adam' (got {gradient_type!r})")
    return [
        python_for(config),
        "-m",
        "less.data_selection.get_info",
        "--info_type",
        "grads",
        "--gradient_type",
        gradient_type,
        "--train_file",
        str(train_files),
        "--model_path",
        str(model_path),
        "--output_path",
        str(output_path),
        "--gradient_projection_dimension",
        str(config.projection_dim),
    ]


# Inline select script. Loads per-checkpoint train/val projected gradients
# from LESS's canonical ``<output>/dim<proj>/all_orig.pt`` layout, computes
# cosine influence (matmul → mean over val rows), averages across
# checkpoints with equal weights, and writes the top-k indices as JSON.
# Replaces ``less.data_selection.matching`` because the upstream script is
# hardcoded to {mmlu, bbh, tydiqa} via N_SUBTASKS.
SELECT_SCRIPT = """
import json
import os
import sys

train_root = sys.argv[1]
val_root = sys.argv[2]
proj_dim = sys.argv[3]
keep = int(sys.argv[4])
output_path = sys.argv[5]

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpts = sorted(d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d)))
if not ckpts:
    raise SystemExit('select: no checkpoint subdirs under ' + train_root)
weight = 1.0 / len(ckpts)

pool_size = None
influence = None
for ckpt in ckpts:
    train_path = os.path.join(train_root, ckpt, 'dim' + proj_dim, 'all_orig.pt')
    val_path = os.path.join(val_root, ckpt, 'dim' + proj_dim, 'all_orig.pt')
    train_g = torch.load(train_path).to(device).float()
    val_g = torch.load(val_path).to(device).float()
    if train_g.dim() != 2:
        raise SystemExit('select: train gradients at ' + train_path + ' must be 2-D, got shape ' + str(tuple(train_g.shape)))
    if val_g.dim() != 2:
        raise SystemExit('select: val gradients at ' + val_path + ' must be 2-D, got shape ' + str(tuple(val_g.shape)))
    if train_g.shape[1] != val_g.shape[1]:
        raise SystemExit('select: projection dim mismatch — train ' + str(train_g.shape[1]) + ' vs val ' + str(val_g.shape[1]))
    if pool_size is None:
        pool_size = train_g.shape[0]
    elif train_g.shape[0] != pool_size:
        raise SystemExit('select: pool size drift across ckpts — first saw ' + str(pool_size) + ', now ' + str(train_g.shape[0]) + ' at ' + train_path)
    score = (train_g @ val_g.T).mean(dim=1)
    influence = score * weight if influence is None else influence + score * weight

if influence is None or influence.dim() != 1 or influence.shape[0] != pool_size:
    raise SystemExit('select: influence vector must be 1-D length ' + str(pool_size) + ', got ' + (str(tuple(influence.shape)) if influence is not None else 'None'))
if keep > pool_size:
    raise SystemExit('select: keep=' + str(keep) + ' exceeds pool size ' + str(pool_size))

ranked = torch.argsort(influence, descending=True)[:keep].tolist()
os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(sorted(int(i) for i in ranked), f)
"""


def select_command(
    config: LessConfig,
    train_grads_root: Path,
    val_grads_root: Path,
    keep: int,
    indices_path: Path,
) -> list[str]:
    """Stage 4 — cosine matching + top-k via inline ``python3 -c`` subprocess."""
    return [
        python_for(config),
        "-c",
        SELECT_SCRIPT,
        str(train_grads_root),
        str(val_grads_root),
        str(config.projection_dim),
        str(keep),
        str(indices_path),
    ]
