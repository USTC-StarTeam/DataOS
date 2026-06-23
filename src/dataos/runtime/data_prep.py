"""Prepare the shared training data ONCE per project (prepare-once model).

All experiments in a study share one prepared dataset, loaded by ``run_one.py --data``
through ``exp.data.set(ds)``. This mirrors the skill's prepare-once doctrine and keeps
the loop's experiments varying only train/eval dimensions.

Returns a path that ``dataos.load`` accepts: a local ``.jsonl`` file (stub) or a
``save_to_disk`` directory (real source). Idempotent — re-running a project reuses the
already-prepared data (resume).
"""

from __future__ import annotations

import json
from pathlib import Path

from dataos.agent.config import AutoRunConfig


def prepare_once(*, cfg: AutoRunConfig, workspace: str | None = None) -> str:
    ws = workspace or cfg.workspace
    # OUTSIDE the project dir (Project.studies lists every project-dir subdir, so a
    # "prepared" dir there would be mis-counted as a study).
    out_dir = Path(ws) / "_prepared" / cfg.project_id

    if cfg.stub or not cfg.data_source:
        path = out_dir / "stub.jsonl"
        if not path.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {"instruction": f"Compute {i} + {i}.", "input": "", "output": str(2 * i)}
                for i in range(8)
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return str(path)

    # real source: load once and save_to_disk (HF arrow + operator sidecar)
    marker = out_dir / "dataset_info.json"
    if marker.exists():
        return str(out_dir)
    import dataos

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = dataos.load(cfg.data_source, limit=cfg.data_limit)
    ds.save_to_disk(str(out_dir))
    return str(out_dir)
