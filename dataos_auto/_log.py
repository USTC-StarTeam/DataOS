"""Run-scoped logging for the autonomous loop.

A long autonomous run is otherwise silent until it returns (the skill's old failure mode:
no trace of where a killed/odd run got to). This installs a human-readable progress log so
every phase — planner I/O, Slurm submit/poll/score, login judge, stop decision — is on disk
and on stderr.

The log file lives **outside** the project dir, at ``<workspace>/_logs/<project_id>/run.log``:
``Project.studies`` counts every subdir of the project dir, so a log dir there would be
mis-counted as a study (same reason ``final_model.json`` is a file, not a subdir).
"""

from __future__ import annotations

import logging
from pathlib import Path

_FMT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_run_logging(workspace: str, project_id: str, level: int = logging.INFO):
    """Configure the ``dataos_auto`` logger to append a run log + stream to stderr.

    Idempotent: re-calling (resume, or several ``run_auto`` calls in one process / the test
    suite) replaces our handlers instead of stacking them. Child loggers (``dataos_auto.*``)
    inherit these handlers. Returns ``(logger, log_path)``."""
    log_dir = Path(workspace) / "_logs" / project_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"

    logger = logging.getLogger("dataos_auto")
    logger.setLevel(level)
    logger.propagate = False  # don't double-log via the root logger
    for h in list(logger.handlers):  # drop our prior handlers (no duplicate lines on re-call)
        if getattr(h, "_dataos_auto", False):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    fmt = logging.Formatter(_FMT, _DATEFMT)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh._dataos_auto = True  # type: ignore[attr-defined]
    sh = logging.StreamHandler()  # stderr — surfaces in console / captured output
    sh.setFormatter(fmt)
    sh._dataos_auto = True  # type: ignore[attr-defined]
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, log_path
