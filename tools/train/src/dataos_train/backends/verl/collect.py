"""verl checkpoint -> HF ``final_model`` merge + metrics parsing.

The merge (memory-heavy) runs INSIDE the job via ``--merge`` (login-node rules forbid heavy
work outside Slurm); ``VerlBackend.collect`` then only parses. ``verl.model_merger`` is
version-sensitive — pinned with the verl env.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_PAIR = re.compile(r"['\"]?([\w/]+)['\"]?\s*[:=]\s*(" + _NUM + r")")
_KEYS = ("reward", "kl", "loss", "response_length", "step", "score")


def latest_global_step(ckpt_root: Path) -> Path | None:
    if not ckpt_root.is_dir():
        return None
    steps: list[tuple[int, Path]] = []
    for d in ckpt_root.iterdir():
        if d.is_dir() and d.name.startswith("global_step_"):
            try:
                steps.append((int(d.name.rsplit("_", 1)[-1]), d))
            except ValueError:
                continue
    return max(steps)[1] if steps else None


def merge_to_hf(run_dir: str) -> str:
    """Consolidate the latest verl FSDP checkpoint into an HF model at ``final_model``."""
    rd = Path(run_dir)
    last = latest_global_step(rd / "verl" / "ckpt")
    if last is None:
        raise FileNotFoundError(f"no global_step_* checkpoint under {rd / 'verl' / 'ckpt'}")
    actor = last / "actor"
    final = rd / "final_model"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(actor),
            "--target_dir",
            str(final),
        ],
        check=True,
    )
    return str(final)


def _candidate_logs(rd: Path) -> list[Path]:
    logs = sorted(rd.glob("slurm-*.out"))
    if (rd / "logs").is_dir():
        logs += sorted((rd / "logs").glob("*.out"))
    return logs


def parse_metrics(run_dir: str) -> dict:
    """Best-effort: prefer a metrics.jsonl, else scrape the last metric line from the logs."""
    rd = Path(run_dir)
    mj = rd / "verl" / "metrics.jsonl"
    if mj.is_file():
        lines = [ln for ln in mj.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            try:
                data = json.loads(lines[-1])
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if isinstance(v, (int, float))}
            except (ValueError, TypeError):
                pass
    best: dict = {}
    for log in _candidate_logs(rd):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(k in line.lower() for k in _KEYS):
                pairs = {k: float(v) for k, v in _PAIR.findall(line)}
                filtered = {k: v for k, v in pairs.items() if any(t in k.lower() for t in _KEYS)}
                if filtered:
                    best = filtered  # keep the last metric-bearing line
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataos_train.backends.verl.collect")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--run-dir", dest="run_dir", required=True)
    args = ap.parse_args(argv)
    if args.merge:
        path = merge_to_hf(args.run_dir)
        print(json.dumps({"final_model": path}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
