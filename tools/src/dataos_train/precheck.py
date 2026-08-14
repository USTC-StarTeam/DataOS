"""Pre-flight checks that need no GPU and no scheduler.

Cheap questions asked before a run is allowed to start, because the answers are expensive to
learn late. Lived in ``cli.py`` until the stage layer landed, where it read as a CLI concern; it
is a property of the spec, so :class:`~dataos_train.stage.TrainStage` asks it in ``validate``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .spec import TrainSpec


#: Free space required before a run is allowed to start, as a multiple of the base model's
#: weight files. A full-SFT run writes back roughly one model (``final_model``); a LoRA run writes
#: an adapter plus a merged export. The headroom covers the export and a little slack. It is
#: still a FLOOR, not a guarantee: nothing here reserves the space against another job filling
#: the same filesystem in the meantime.
_DISK_HEADROOM = 1.5


def _retained_models(spec: TrainSpec) -> float:
    """How many model-sized artifacts this run keeps on disk.

    Trajectory checkpoints (``save_steps``) each cost one model's worth — ``save_only_model`` is
    forced in the rendered config, so no optimizer state — but only for full/freeze tuning. A
    LoRA checkpoint is an adapter, orders of magnitude smaller, so it does not move this number.
    """
    method = spec.method
    if not getattr(method, "save_steps", None):
        return _DISK_HEADROOM
    if getattr(method, "finetuning_type", None) == "lora":
        return _DISK_HEADROOM
    return _DISK_HEADROOM + int(getattr(method, "save_total_limit", 0) or 0)


def _model_bytes(path: str) -> int:
    """Size of the model's weight files, or 0 when the path is not a readable model dir."""
    root = Path(path)
    if not root.is_dir():
        return 0
    return sum(
        f.stat().st_size
        for f in root.iterdir()
        if f.is_file() and f.suffix in {".safetensors", ".bin", ".pt"}
    )


def precheck_disk(spec: TrainSpec) -> list[str]:
    """Refuse a run that cannot fit its own output.

    The 4b2 v5 batch exhausted ``/data2`` mid-run: training finished, then ``save_pretrained``
    died writing the weights, so the GPU hours were spent and nothing came back. The signature is
    unhelpful too — a full filesystem surfaces as safetensors' "File too large (os error 27)",
    which reads like a size bug rather than a space one. Checking before submit turns hours of
    wasted compute into one line of output.
    """
    factor = _retained_models(spec)
    need = int(_model_bytes(spec.base_model) * factor)
    if not need:  # unreadable base (e.g. a spec built off-cluster) — nothing to compare against
        return []
    probe = Path(spec.run_dir)
    while not probe.exists() and probe != probe.parent:  # run_dir is created by the job
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free >= need:
        return []
    gb = 1024**3
    keep = getattr(spec.method, "save_steps", None) and getattr(spec.method, "save_total_limit", 0)
    return [
        f"not enough free space under {spec.run_dir}: {free / gb:.1f} GiB free, this run needs "
        f"at least {need / gb:.1f} GiB ({factor}x the base model's "
        f"{_model_bytes(spec.base_model) / gb:.1f} GiB of weights) to write its output"
        + (f", including {keep} retained checkpoints" if keep else "")
        + ". Free space, lower save_total_limit, or point run_dir at another filesystem — "
        "training to completion and then failing on save wastes the whole allocation."
    ]
