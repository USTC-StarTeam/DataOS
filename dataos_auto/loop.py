"""The deterministic autonomous loop — the part the skill used to ask a human/Claude
to perform by hand.

Everything here is owned Python: the round loop, the fan-out, the stop decision, and
resume. The only LLM calls are ``planner.next_plan`` and ``reviewer.conclude``. Stopping
is a single wall-clock condition (10h) in :mod:`dataos_auto.budget` (the LLM never stops the loop).

Resume is project-scoped and crash-safe: a round is "done" iff its study has a
review.json; the loop restarts at the first round without one, reuses any plan already
written, and the executor's ``skip_completed`` reruns only unfinished experiments.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from . import budget as _budget
from . import executor as _executor
from .config import AutoRunConfig
from .data_prep import prepare_once
from .eval_freeze import freeze_eval

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


def _safe_catalog() -> list[dict]:
    try:
        import dataos

        return dataos.catalog()
    except Exception:
        return []


def _better(a: dict | None, b: dict | None) -> dict | None:
    """Return whichever of two best-dicts has the higher score (None-safe)."""
    av = a.get("score") if a else None
    bv = b.get("score") if b else None
    if not isinstance(bv, (int, float)):
        return a
    if not isinstance(av, (int, float)):
        return b
    return a if av >= bv else b


def _reviewed_studies(proj) -> list[str]:
    return sorted(
        sid for sid in proj.studies if (Path(proj.project_dir) / sid / "review.json").is_file()
    )


def _global_best(proj) -> dict | None:
    best = None
    for s in proj.summary().get("studies", []):
        best = _better(best, s.get("best"))
    return best


def _best_model_over_project(proj):
    """The historically BEST completed experiment across the whole project, as
    (score, study_id, exp_name, model_path). The autonomous run declares THIS as its
    final_model — not whatever the last round produced."""
    best = None
    for sid in proj.studies:
        edir = Path(proj.project_dir) / sid / "experiments"
        if not edir.is_dir():
            continue
        for slug in sorted(p.name for p in edir.iterdir() if p.is_dir()):
            try:
                rep = json.loads((edir / slug / "report.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            score = rep.get("score")
            if not isinstance(score, (int, float)):
                continue
            cand = (float(score), sid, rep.get("name", slug), (rep.get("model") or {}).get("path"))
            if best is None or cand[0] > best[0]:
                best = cand
    return best


def _resume_context(proj):
    """Rebuild (prev_digest, prev_review) from the last reviewed study, for the planner."""
    from .schemas import ReviewOut

    reviewed = _reviewed_studies(proj)
    if not reviewed:
        return None, None
    sid = reviewed[-1]
    digest = proj.study(sid).summary()
    review = None
    try:
        rj = json.loads((Path(proj.project_dir) / sid / "review.json").read_text(encoding="utf-8"))
        c = rj.get("conclusion")
        if isinstance(c, dict):
            review = ReviewOut(conclusion=c.get("conclusion", ""), direction=c.get("direction", ""))
        elif isinstance(c, str):
            review = ReviewOut(conclusion=c, direction="")
    except Exception:
        pass
    return digest, review


def _finish(proj, reason: str, best: dict | None, rounds: int, events: list) -> dict:
    # Declare the BEST model over all rounds (not the last) as the final_model, and persist
    # a PostTrainBench-style status/final_model.json pointing at it.
    from dataos._ids import _utc_now_iso
    from dataos._io import atomic_write_json

    chosen = _best_model_over_project(proj)
    final_model = None
    if chosen:
        score, sid, name, mpath = chosen
        final_model = {
            "final_model": mpath,
            "score": score,
            "study": sid,
            "experiment": name,
            "source": "best_over_history",
            "declared_at": _utc_now_iso(),
        }
        # a FILE at the project root (not a subdir — Project.studies counts every subdir)
        atomic_write_json(Path(proj.project_dir) / "final_model.json", final_model)
    return {
        "project_id": proj.id,
        "stop_reason": reason,
        "rounds_completed": rounds,
        "best": best,
        "final_model": final_model,
        "events": events,
        "summary": proj.summary(),
    }


def run_auto(cfg: AutoRunConfig, *, _clock=time.monotonic) -> dict[str, Any]:
    """Drive a project until the wall-clock budget is spent, then return a result dict.

    Single stop reason: ``budget:wall_clock`` (default 10h, the PTB protocol). ``_clock``
    is an injectable monotonic clock for deterministic tests."""
    import dataos

    from .planner import next_plan
    from .reviewer import conclude
    from .backend import make_agent
    from ._log import setup_run_logging

    cfg.workspace = os.path.abspath(cfg.workspace)
    log, log_path = setup_run_logging(cfg.workspace, cfg.project_id)
    log.info(
        "run start: project=%s target_bench=%s executor=%s backend=%s profile=%s wall_budget=%ss",
        cfg.project_id, cfg.target_bench, cfg.train_executor, cfg.backend,
        cfg.ptb_profile, cfg.budget.max_wall_clock_s,
    )
    log.info("run log: %s", log_path)

    proj = dataos.project(id=cfg.project_id, workspace=cfg.workspace)
    proj.set_goal(
        goal=cfg.goal or None,
        instruct=cfg.instruct or None,
        mode="autonomous",
        success_criterion=cfg.criterion.as_dict(),
        budget=cfg.budget.as_dict(),
    )
    proj.load()  # rehydrate prior studies/experiments (no-op on first run)

    eval_cfg = freeze_eval(cfg=cfg)
    # slurm_h100 prepares data inside the Slurm job (from the fixed pool on H100); only the
    # local executor needs a prepared data dir here.
    data_dir = None if cfg.train_executor == "slurm_h100" else prepare_once(cfg=cfg)
    catalog = _safe_catalog()

    started = _clock()
    round_idx = len(_reviewed_studies(proj))
    best = _global_best(proj)
    prev_digest, prev_review = _resume_context(proj)
    events: list[dict] = []
    if round_idx:
        log.info("resuming: %d reviewed round(s) on disk; continuing at round %d (best=%s)",
                 round_idx, round_idx, best)

    while True:
        reason = _budget.stop_reason(started_at=started, budget=cfg.budget, now=_clock())
        if reason:
            log.info("STOP: %s (after %d round(s))", reason, round_idx)
            result = _finish(proj, reason, best, round_idx, events)
            log.info("final_model: %s", result.get("final_model"))
            return result

        study_id = f"round-{round_idx:04d}"
        study = proj.study(study_id)

        log.info("round %d: start (study=%s)", round_idx, study_id)
        # reuse a plan already written for this round (crash mid-round), else ask the planner
        if study.plan and study.plan.get("experiments"):
            specs = study.plan["experiments"]
            log.info("round %d: reusing on-disk plan (%d experiment(s))", round_idx, len(specs))
        else:
            log.info("round %d: planning via %s backend...", round_idx, cfg.backend)
            plan = next_plan(
                agent=make_agent("planner", cfg),
                instruct=cfg.instruct,
                goal=cfg.goal,
                prev_review=prev_review,
                prev_digest=prev_digest,
                eval_cfg=eval_cfg,
                round_idx=round_idx,
                catalog=catalog,
            )
            specs = []
            for s in plan.experiments:
                d = s.model_dump()
                d["eval"] = dict(eval_cfg)  # freeze: the planner may not touch the eval protocol
                if cfg.train_executor == "slurm_h100":
                    # ship training to H100; pin the base model so the planner only sweeps
                    # hyperparams (consistent base across the study).
                    tr = dict(d.get("train") or {})
                    tr["executor"] = "slurm_h100"
                    if cfg.slurm:
                        tr["slurm"] = cfg.slurm
                    if cfg.h100_base_model:
                        tr["model_name_or_path"] = cfg.h100_base_model
                    if cfg.base_model_id:
                        tr["base_model_id"] = cfg.base_model_id
                    d["train"] = tr
                specs.append(d)
            study.set_plan({"goal": plan.goal, "experiments": specs})
            specs = study.plan["experiments"]
            names = [s.get("name") for s in specs]
            log.info("round %d: plan goal=%r -> %d experiment(s): %s",
                     round_idx, plan.goal, len(specs), names)

        log.info("round %d: executing %d spec(s) via %s", round_idx, len(specs), cfg.train_executor)
        if cfg.train_executor == "slurm_h100":
            from . import h100_slurm

            h100_slurm.fan_out(cfg=cfg, study_id=study_id, specs=specs)
        else:
            _executor.fan_out(
                project_id=cfg.project_id,
                study_id=study_id,
                workspace=cfg.workspace,
                specs=specs,
                data_dir=data_dir,
                gpus=cfg.gpus,
                run_one=cfg.run_one,
                cwd=_REPO_ROOT,
                poll_interval_s=cfg.poll_interval_s,
                skip_completed=True,
                limit=cfg.data_limit,
            )

        digest = study.summary()
        log.info("round %d: results best=%s n_experiments=%s caution=%s",
                 round_idx, digest.get("best"), digest.get("n_experiments"), digest.get("caution"))
        log.info("round %d: reviewing...", round_idx)
        review = conclude(
            agent=make_agent("reviewer", cfg), digest=digest, goal=cfg.goal, round_idx=round_idx
        )
        study.review(conclusion={"conclusion": review.conclusion, "direction": review.direction})
        log.info("round %d: review conclusion=%r direction=%r",
                 round_idx, (review.conclusion or "")[:160], (review.direction or "")[:160])

        best = _better(best, digest.get("best"))
        prev_digest, prev_review = digest, review
        events.append(
            {
                "round": round_idx,
                "study": study_id,
                "best": digest.get("best"),
                "n_experiments": digest.get("n_experiments"),
                "caution": digest.get("caution"),
            }
        )
        round_idx += 1
