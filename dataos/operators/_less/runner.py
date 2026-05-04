from __future__ import annotations

import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from dataos.core.sandbox import Sandbox
from dataos.operators._less.cache import (
    CacheEntry,
    make_cache_entry,
    resolve_cache_root,
    train_grads_key,
    val_grads_key,
    warmup_key,
)
from dataos.operators._less.commands import (
    grad_command,
    select_command,
    warmup_command,
)
from dataos.operators._less.config import LessConfig
from dataos.operators._less.io import export_jsonl  # noqa: F401  (kept for future generic uses)

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


# ── Stub mode ────────────────────────────────────────────────────────────────


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


# ── Real mode ────────────────────────────────────────────────────────────────


def run_less_pipeline(
    config: LessConfig,
    pool_artifact_id: str,
    pool_dataset: Dataset,
    target_artifact_id: str,
    target_dataset: Dataset,
    sandbox: Sandbox,
) -> PipelineResult:
    """Real LESS pipeline: 4 stages via subprocess, content-addressed cache.

    Stages (each with cache key + cache hit short-circuit, except select):
      1. warmup       — LoRA training on the candidate pool
      2. train_grads  — projected SGD gradients per warmup checkpoint × pool
      3. val_grads    — projected Adam gradients per warmup checkpoint × target
      4. select       — cosine matching → top-k indices (always recomputed)
    """
    repo_path = Path(config.less_repo_path).expanduser()
    if not repo_path.exists():
        raise ValueError(f"less: less_repo_path does not exist: {repo_path}")
    if shutil.which("python3") is None:
        raise RuntimeError("less: 'python3' executable not found on PATH")
    if shutil.which("torchrun") is None:
        raise RuntimeError(
            "less: 'torchrun' not found on PATH (required for warmup); "
            "install torch in the active environment"
        )

    cache_root = resolve_cache_root(sandbox.root, config)
    cache_root.mkdir(parents=True, exist_ok=True)

    work_dir = sandbox.tmp_dir / "less"
    work_dir.mkdir(parents=True, exist_ok=True)
    pool_jsonl = work_dir / "pool.jsonl"
    target_jsonl = work_dir / "target.jsonl"
    _export_for_less(pool_dataset, pool_jsonl)
    _export_for_less(target_dataset, target_jsonl)

    keys = {
        "warmup": warmup_key(pool_artifact_id, config),
    }
    keys["train_grads"] = train_grads_key(keys["warmup"], config)
    keys["val_grads"] = val_grads_key(keys["warmup"], target_artifact_id, config)

    stages: list[StageResult] = []

    # ── Stage 1: warmup ──────────────────────────────────────────────────────
    warmup_entry = make_cache_entry(cache_root, "warmup", keys["warmup"])
    warmup_log = sandbox.logs_dir / "less_warmup.log"
    if not config.force_recompute and warmup_entry.hit():
        stages.append(_hit(warmup_entry, "warmup"))
    else:
        warmup_dir = warmup_entry.reserve()
        _run(
            warmup_command(config, pool_jsonl, warmup_dir),
            cwd=repo_path,
            log_path=warmup_log,
            header=True,
        )
        ckpts = sorted(
            p for p in warmup_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")
        )
        if not ckpts:
            raise RuntimeError(
                f"less: warmup produced no checkpoint-* directories in {warmup_dir} "
                f"(see {warmup_log})"
            )
        warmup_entry.write_meta(
            {"checkpoints": [str(p) for p in ckpts], "log_path": str(warmup_log)}
        )
        stages.append(
            StageResult(
                name="warmup",
                cache_status="miss",
                cache_key=keys["warmup"],
                cache_path=str(warmup_dir),
                log_path=str(warmup_log),
            )
        )

    checkpoints = [Path(p) for p in warmup_entry.read_meta()["checkpoints"]]
    _ensure_less_compatible_optimizer(checkpoints, config.base_model)

    # ── Stage 2: train data gradients ────────────────────────────────────────
    tg_entry = make_cache_entry(cache_root, "train_grads", keys["train_grads"])
    tg_log = sandbox.logs_dir / "less_train_grads.log"
    if not config.force_recompute and tg_entry.hit():
        stages.append(_hit(tg_entry, "train_grads"))
    else:
        tg_dir = tg_entry.reserve()
        _run_grads_per_checkpoint(
            config=config,
            gradient_type="adam",
            train_files=pool_jsonl,
            checkpoints=checkpoints,
            output_root=tg_dir,
            cwd=repo_path,
            log_path=tg_log,
            stage_label="train_grads",
        )
        tg_entry.write_meta(
            {
                "grads_dir": str(tg_dir),
                "checkpoints": [str(p) for p in checkpoints],
                "log_path": str(tg_log),
            }
        )
        stages.append(
            StageResult(
                name="train_grads",
                cache_status="miss",
                cache_key=keys["train_grads"],
                cache_path=str(tg_dir),
                log_path=str(tg_log),
            )
        )

    # ── Stage 3: validation (target) gradients ───────────────────────────────
    vg_entry = make_cache_entry(cache_root, "val_grads", keys["val_grads"])
    vg_log = sandbox.logs_dir / "less_val_grads.log"
    if not config.force_recompute and vg_entry.hit():
        stages.append(_hit(vg_entry, "val_grads"))
    else:
        vg_dir = vg_entry.reserve()
        _run_grads_per_checkpoint(
            config=config,
            gradient_type="sgd",
            train_files=target_jsonl,
            checkpoints=checkpoints,
            output_root=vg_dir,
            cwd=repo_path,
            log_path=vg_log,
            stage_label="val_grads",
        )
        vg_entry.write_meta(
            {
                "grads_dir": str(vg_dir),
                "checkpoints": [str(p) for p in checkpoints],
                "log_path": str(vg_log),
            }
        )
        stages.append(
            StageResult(
                name="val_grads",
                cache_status="miss",
                cache_key=keys["val_grads"],
                cache_path=str(vg_dir),
                log_path=str(vg_log),
            )
        )

    # ── Stage 4: cosine matching + top-k (always runs) ───────────────────────
    select_dir = work_dir / "select"
    select_dir.mkdir(parents=True, exist_ok=True)
    select_log = sandbox.logs_dir / "less_select.log"
    indices_path = select_dir / "selected_indices.json"
    keep = _resolve_keep_count(config, pool_dataset.num_rows)

    _run(
        select_command(
            config,
            train_grads_root=tg_entry.path,
            val_grads_root=vg_entry.path,
            keep=keep,
            indices_path=indices_path,
        ),
        cwd=repo_path,
        log_path=select_log,
        header=True,
    )
    if not indices_path.exists():
        raise RuntimeError(
            f"less: select stage did not produce {indices_path} (see {select_log})"
        )

    raw_indices = json.loads(indices_path.read_text(encoding="utf-8"))
    if not isinstance(raw_indices, list) or not all(isinstance(item, int) for item in raw_indices):
        raise RuntimeError(
            f"less: selected_indices.json must be a list of ints (got {type(raw_indices).__name__})"
        )

    stages.append(
        StageResult(
            name="select",
            cache_status="miss",
            cache_key=None,
            cache_path=str(select_dir),
            log_path=str(select_log),
        )
    )

    return PipelineResult(
        selected_indices=raw_indices,
        stages=stages,
        cache_root=cache_root,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_keep_count(config: LessConfig, pool_size: int) -> int:
    if config.top_k is not None:
        if config.top_k > pool_size:
            raise ValueError(f"less: top_k={config.top_k} exceeds pool size {pool_size}")
        return config.top_k
    fraction = config.top_fraction if config.top_fraction is not None else 0.0
    keep = max(1, int(round(pool_size * fraction)))
    return min(keep, pool_size)


def _ensure_less_compatible_optimizer(checkpoints: list[Path], base_model: str) -> None:
    """Rewrite each checkpoint's optimizer state into the format LESS expects.

    Bridges two upstream changes between LESS (pinned to ``transformers==4.36.2``)
    and the modern stack we run on:

    1. Filename: transformers now saves ``optimizer.pt``; LESS's
       ``data_selection/get_info.py`` does ``torch.load(<ckpt>/optimizer.bin)``.
    2. Schema: modern PyTorch's ``optimizer.state_dict()['state']`` is keyed
       by integer parameter position, but LESS's ``prepare_optimizer_state``
       indexes it by parameter name (``optimizer_state[name]['exp_avg']``).
       That works only if the dict is name-keyed.

    We instantiate the PEFT adapter at each checkpoint to read the trainable
    parameter names in ``named_parameters()`` order, zip them with the
    optimizer's per-group parameter ids, and save a name-keyed dict under
    ``optimizer.bin`` next to the original ``optimizer.pt``. Idempotent.
    """
    pending: list[Path] = []
    for ckpt in checkpoints:
        modern = ckpt / "optimizer.pt"
        legacy = ckpt / "optimizer.bin"
        if not modern.exists():
            continue
        if legacy.exists() and not legacy.is_symlink() and legacy.stat().st_size > 1024:
            continue
        if legacy.is_symlink():
            legacy.unlink()
        pending.append(ckpt)
    if not pending:
        return

    # Heavy imports deferred until we actually need to load the model.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_loaded = None
    try:
        for ckpt in pending:
            legacy = ckpt / "optimizer.bin"
            modern = ckpt / "optimizer.pt"

            state = torch.load(str(modern), map_location="cpu", weights_only=False)
            int_state = state.get("state", {})
            ids: list[int] = []
            for group in state.get("param_groups", []):
                ids.extend(group.get("params", []))

            if base_loaded is None:
                base_loaded = AutoModelForCausalLM.from_pretrained(
                    base_model, torch_dtype="auto"
                )
            model = PeftModel.from_pretrained(base_loaded, str(ckpt))
            # Mirror LESS's get_info.load_model: mark LoRA params trainable so
            # `named_parameters()` filtered by requires_grad matches the names
            # the optimizer was constructed with.
            for n, p in model.named_parameters():
                if "lora" in n.lower():
                    p.requires_grad = True
            names = [n for n, p in model.named_parameters() if p.requires_grad]

            if len(names) != len(ids):
                raise RuntimeError(
                    f"less: optimizer state has {len(ids)} params but model has "
                    f"{len(names)} trainable params at {ckpt}"
                )
            name_keyed = {
                name: int_state[pid] for name, pid in zip(names, ids) if pid in int_state
            }
            torch.save(
                {"state": name_keyed, "param_groups": state.get("param_groups", [])},
                str(legacy),
            )
            # Drop the adapter overlay so the next checkpoint starts from the base.
            base_loaded = model.unload() if hasattr(model, "unload") else base_loaded
    finally:
        del base_loaded


def _export_for_less(dataset: "Dataset", target: Path) -> None:
    """Dump rows in the schema LESS's loader expects.

    LESS's ``get_training_dataset.py`` requires either ``prompt``/``completion``
    or a chat-format ``messages`` column. DataOS's canonical schema is
    ``instruction``/``response`` (+ optional ``input``/``system``), so we emit
    a ``messages`` column on the fly. Optional ``system`` becomes the leading
    system turn; ``input`` is concatenated into the user content. This keeps
    the LESS-specific massaging local to the operator rather than polluting
    the generic ``io.export_jsonl`` helper.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        columns = set(dataset.column_names)
        for row in dataset:
            user_content = str(row.get("instruction", "") or "")
            extra = row.get("input") if "input" in columns else None
            if extra:
                user_content = f"{user_content}\n\n{extra}".strip()
            messages = []
            if "system" in columns and row.get("system"):
                messages.append({"role": "system", "content": str(row["system"])})
            messages.append({"role": "user", "content": user_content})
            messages.append(
                {"role": "assistant", "content": str(row.get("response", "") or "")}
            )
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")


def _hit(entry: CacheEntry, stage: str) -> StageResult:
    meta = entry.read_meta()
    return StageResult(
        name=stage,
        cache_status="hit",
        cache_key=entry.key,
        cache_path=str(entry.path),
        log_path=meta.get("log_path"),
    )


def _run_grads_per_checkpoint(
    *,
    config: LessConfig,
    gradient_type: str,
    train_files: Path,
    checkpoints: list[Path],
    output_root: Path,
    cwd: Path,
    log_path: Path,
    stage_label: str,
) -> None:
    for index, ckpt in enumerate(checkpoints):
        per_ckpt_dir = output_root / ckpt.name
        per_ckpt_dir.mkdir(parents=True, exist_ok=True)
        _run(
            grad_command(
                config,
                gradient_type=gradient_type,
                train_files=train_files,
                model_path=ckpt,
                output_path=per_ckpt_dir,
            ),
            cwd=cwd,
            log_path=log_path,
            header=(index == 0),
        )
        if not any(per_ckpt_dir.iterdir()):
            raise RuntimeError(
                f"less: {stage_label} for {ckpt.name} produced no output files in "
                f"{per_ckpt_dir} (see {log_path})"
            )


def _run(args: list[str], cwd: Path, log_path: Path, header: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if header else "a"
    with log_path.open(mode, encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(args) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"less: subprocess failed with exit code {completed.returncode} "
            f"(command={' '.join(args)}, see {log_path})"
        )
