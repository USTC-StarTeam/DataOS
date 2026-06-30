"""Agentic autonomous loop.

The old main loop exposed the tree and deterministically picked three nodes per
round. The current loop is simpler and more agentic: Python maintains durable
state; one persistent main-agent dialog proposes the next single experiment plan;
Python validates/applies it; one sub-agent executes data/train/eval; the main
dialog reviews the report; Python writes the tree.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from dataos.core.io import atomic_write_json
from dataos.agent.eval_freeze import freeze_eval

from . import budget as _budget
from .config import AutoRunConfig

_MAIN_SESSION_FILE = "main_agent_session.json"
_PLAN_RETRIES = 3
_HEARTBEAT_S = 20  # how often each run rewrites its live.json heartbeat


def _monitor_port(cfg) -> int:
    """Live-monitor port: ``DATAOS_MONITOR_PORT`` overrides the config; ``0`` disables."""
    raw = os.environ.get("DATAOS_MONITOR_PORT")
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            pass
    return cfg.monitor_port


def _start_monitor(cfg, log) -> None:
    """Persist this run's heartbeat, and host the shared live monitor when the port is
    free. ONE monitor serves every run under the workspace (it reads each run's tree.json
    + live.json), so a single port is enough; if its host dies, another run takes over on
    the next retry. ``DATAOS_MONITOR_PORT=0`` (or config 0) disables hosting but still
    writes the heartbeat, so the standalone monitor can show this run."""
    from dataos.agent import monitor

    monitor.configure((cfg.runtime_config or {}).get("cluster"))
    workspace = Path(cfg.workspace)

    def heartbeat():
        while True:
            time.sleep(_HEARTBEAT_S)
            monitor.write_live(workspace, cfg.project_id)

    threading.Thread(target=heartbeat, name="dataos-heartbeat", daemon=True).start()

    port = _monitor_port(cfg)
    if port <= 0:
        return

    def supervise():
        while True:
            try:
                monitor.serve_in_background(workspace, port=port)
                log.info("monitor: hosting live view at http://127.0.0.1:%d", port)
                return  # we hold the port; its daemon server thread serves until exit
            except OSError:
                time.sleep(30)  # another run hosts it; retry so we take over if it dies

    threading.Thread(target=supervise, name="dataos-monitor-sup", daemon=True).start()


def _publish_live(cfg, *, round_idx: int, phase: str, started: float, now: float, **extra) -> None:
    """Publish the loop's current phase, and persist it to live.json (best-effort)."""
    from dataos.agent import monitor

    cap = cfg.budget.max_wall_clock_s
    monitor.LIVE.publish(
        run_id=cfg.project_id,
        bench=cfg.target_bench,
        round=round_idx,
        phase=phase,
        remaining_hours=_remaining_hours(started, cfg, now),
        budget_hours=(cap / 3600.0) if cap else None,
        **extra,
    )
    monitor.write_live(Path(cfg.workspace), cfg.project_id)


def _safe_catalog() -> list[dict]:
    try:
        import dataos

        return dataos.catalog()
    except Exception:
        return []


def _better(a: dict | None, b: dict | None) -> dict | None:
    av = a.get("score") if a else None
    bv = b.get("score") if b else None
    if not isinstance(bv, (int, float)):
        return a
    if not isinstance(av, (int, float)):
        return b
    return a if av >= bv else b


def _report_score(report: dict) -> float | None:
    score = report.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    evaluation = report.get("eval") or {}
    macro = evaluation.get("macro_average")
    if isinstance(macro, (int, float)):
        return float(macro)
    score = evaluation.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _read_report(exp_root: Path, name: str):
    """Read one experiment report as score/model/artifacts/scripts/status."""
    from dataos.core.study import _slug

    path = exp_root / _slug(name) / "report.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None, None, {}, {}, None, str(path)
    model = report.get("model") or {}
    return (
        _report_score(report),
        model.get("path"),
        report.get("artifacts") or {},
        report.get("scripts") or {},
        report.get("status"),
        str(path),
    )


def _reviewed_studies(proj) -> list[str]:
    return sorted(
        sid for sid in proj.studies if (Path(proj.project_dir) / sid / "review.json").is_file()
    )


def _global_best(proj) -> dict | None:
    best = None
    for study in proj.summary().get("studies", []):
        best = _better(best, study.get("best"))
    return best


def _best_model_over_project(proj):
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
            score = _report_score(rep)
            if score is None:
                continue
            cand = (score, sid, rep.get("name", slug), (rep.get("model") or {}).get("path"))
            if best is None or cand[0] > best[0]:
                best = cand
    return best


def _load_session(project_dir: Path) -> str | None:
    path = project_dir / _MAIN_SESSION_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    sid = data.get("session_id")
    return sid if isinstance(sid, str) and sid else None


def _save_session(project_dir: Path, *, session_id: str | None, phase: str, round_idx: int) -> None:
    if not session_id:
        return
    atomic_write_json(
        project_dir / _MAIN_SESSION_FILE,
        {
            "session_id": session_id,
            "phase": phase,
            "round": round_idx,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def _recent_reviews(proj, *, limit: int = 5) -> list[dict]:
    rows: list[dict] = []
    for sid in _reviewed_studies(proj)[-limit:]:
        path = Path(proj.project_dir) / sid / "review.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        rows.append(
            {
                "study_id": sid,
                "best": data.get("best"),
                "conclusion": data.get("conclusion"),
                "caution": data.get("caution"),
            }
        )
    return rows


def _remaining_hours(started: float, cfg: AutoRunConfig, now: float) -> float | None:
    cap = cfg.budget.max_wall_clock_s
    if cap is None:
        return None
    return max(0.0, (cap - (now - started)) / 3600.0)


def _effective_eval_limit(cfg: AutoRunConfig) -> int:
    if cfg.eval_limit:
        return cfg.eval_limit
    return 20 if cfg.ptb_profile == "smoke" else 200


def _plan_context(
    *,
    cfg: AutoRunConfig,
    tree,
    proj,
    round_idx: int,
    eval_cfg: dict,
    started: float,
    now: float,
) -> dict:
    return {
        "setting": {
            "project_id": cfg.project_id,
            "goal": cfg.goal or cfg.instruct,
            "scenario": cfg.scenario,
            "target_bench": cfg.target_bench,
            "ptb_profile": cfg.ptb_profile,
            "runtime_config": cfg.runtime_config_path,
        },
        "budget": {
            "max_wall_clock_s": cfg.budget.max_wall_clock_s,
            "remaining_hours": _remaining_hours(started, cfg, now),
        },
        "round_idx": round_idx,
        "tree": tree.to_context(),
        "best": _global_best(proj),
        "recent_reviews": _recent_reviews(proj),
        "frozen_eval": eval_cfg,
        "eval_policy": {
            "mode": "sampled_loop_eval",
            "limit": _effective_eval_limit(cfg),
            "note": "The search loop uses sampled PTB eval for fast ranking; do not plan full eval.",
        },
        "fixed_pool": {
            "path": (cfg.runtime_config.get("paths") or {}).get("fixed_pool"),
            # the pool lives on the H100 cluster — the main-agent ssh's in and explores it itself
            "cluster": {
                k: (cfg.runtime_config.get("cluster") or {}).get(k)
                for k in ("host", "user", "port", "key")
            },
        },
        "constraints": [
            "Return exactly one ExperimentPlan; no action/stop field.",
            "parent_node_id must be one of tree.growable_node_ids.",
            "Child nodes continue training from the selected parent's output model.",
            "data.sources must be real source_ids you verified by exploring fixed_pool.path on "
            "the cluster — never invent names or use placeholders.",
            "Do not set train.model_name_or_path; code injects the parent model path.",
            "One sub-agent owns the full data/train/eval experiment.",
            "Use the frozen eval protocol for this run (eval_limit from runtime config; -1 = full).",
            "GPU work must run through the configured executor, not on a login node.",
        ],
    }


def _validate_plan(plan, tree, cfg: AutoRunConfig) -> list[str]:
    errors: list[str] = []
    parent = plan.parent_node_id
    if not tree.has_node(parent):
        errors.append(f"parent_node_id {parent!r} does not exist")
    elif parent not in {node["id"] for node in tree.growable_nodes()}:
        errors.append(f"parent_node_id {parent!r} is not growable because it has no model")

    name = (plan.name or "").strip()
    if not name:
        errors.append("name is required")
    elif name in tree.experiment_names():
        errors.append(f"name {name!r} already exists in the tree")

    if "model_name_or_path" in (plan.train or {}):
        errors.append("train.model_name_or_path must be omitted; code injects the parent model")
    if plan.estimated_hours is not None and plan.estimated_hours <= 0:
        errors.append("estimated_hours must be positive when provided")
    return errors


def _spec_from_node(node: dict, eval_cfg: dict) -> dict:
    plan = dict(node.get("plan") or {})
    train = dict(plan.get("train") or {})
    init_model = node.get("init_model_path")
    if init_model:
        train["model_name_or_path"] = init_model
    return {
        "name": node["experiment_name"],
        "node_id": node["id"],
        "parent_node_id": node.get("parent"),
        "init_model": init_model,
        "data": dict(plan.get("data") or {}),
        "train": train,
        "eval": dict(eval_cfg),
        "hypothesis": plan.get("hypothesis", ""),
        "expected_signal": plan.get("expected_signal", ""),
        "subagent_brief": plan.get("subagent_brief", ""),
    }


def _finish(proj, reason: str, best: dict | None, rounds: int, events: list, tree=None) -> dict:
    from dataos.core.ids import _utc_now_iso

    final_model = None
    node = tree.best_node() if tree is not None else None
    if node is not None:
        final_model = {
            "final_model": node.get("model_path"),
            "score": node.get("score"),
            "node": node.get("id"),
            "depth": node.get("depth"),
            "source": "best_over_tree",
            "declared_at": _utc_now_iso(),
        }
    else:
        chosen = _best_model_over_project(proj)
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
    if final_model:
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
    """Run the agentic tree loop until the wall-clock budget is spent."""

    import dataos

    from .backend import make_agent
    from .logging import setup_run_logging
    from .planner import next_plan
    from .reviewer import review_result
    from .state_tree import TreeState

    cfg.workspace = os.path.abspath(cfg.workspace)
    log, log_path = setup_run_logging(cfg.workspace, cfg.project_id)
    log.info(
        "run start: project=%s bench=%s backend=%s profile=%s wall_budget=%ss",
        cfg.project_id,
        cfg.target_bench,
        cfg.backend,
        cfg.ptb_profile,
        cfg.budget.max_wall_clock_s,
    )
    log.info("run log: %s", log_path)

    proj = dataos.project(id=cfg.project_id, workspace=cfg.workspace)
    proj.set_goal(
        goal=cfg.goal or None,
        instruct=cfg.instruct or None,
        mode="agentic_main_dialog",
        success_criterion=cfg.criterion.as_dict(),
        budget=cfg.budget.as_dict(),
    )
    proj.load()

    eval_cfg = freeze_eval(cfg=cfg)
    catalog = _safe_catalog()
    base_model = cfg.h100_base_model or cfg.base_model_id or "base"
    tree = TreeState.load(proj.project_dir, base_model)

    main_agent = make_agent("main_agent", cfg)
    session_id = _load_session(Path(proj.project_dir))
    if session_id:
        log.info("main-agent: resuming dialog session %s", session_id)

    started = _clock()
    round_idx = len(_reviewed_studies(proj))
    best = _global_best(proj)
    events: list[dict] = []
    if round_idx:
        log.info(
            "resuming: %d reviewed node(s) on disk; continuing at round %d (best=%s)",
            round_idx,
            round_idx,
            best,
        )

    _start_monitor(cfg, log)

    while True:
        now = _clock()
        reason = _budget.stop_reason(started_at=started, budget=cfg.budget, now=now)
        if reason:
            log.info("STOP: %s (after %d completed node(s))", reason, round_idx)
            _publish_live(cfg, round_idx=round_idx, phase="done", started=started, now=now)
            result = _finish(proj, reason, best, round_idx, events, tree)
            log.info("final_model: %s", result.get("final_model"))
            return result

        study_id = f"round-{round_idx:04d}"
        study = proj.study(study_id)
        log.info("round %d: start (study=%s)", round_idx, study_id)

        node_ids = tree.nodes_for_round(round_idx)
        if len(node_ids) > 1:
            raise RuntimeError(f"round {round_idx} has multiple tree nodes: {node_ids}")

        if node_ids:
            node_id = node_ids[0]
            node = tree.get(node_id)
            log.info(
                "round %d: reusing node %s parent=%s status=%s",
                round_idx,
                node_id,
                node.get("parent"),
                node.get("status"),
            )
        else:
            _publish_live(
                cfg, round_idx=round_idx, phase="planning", started=started, now=now,
                node_id=None, node_name=None, node_run_root=None,
            )
            validation_errors: list[str] | None = None
            plan = None
            for attempt in range(1, _PLAN_RETRIES + 1):
                context = _plan_context(
                    cfg=cfg,
                    tree=tree,
                    proj=proj,
                    round_idx=round_idx,
                    eval_cfg=eval_cfg,
                    started=started,
                    now=now,
                )
                log.info("round %d: asking main-agent for plan (attempt %d)", round_idx, attempt)
                plan, session_id = next_plan(
                    agent=main_agent,
                    context=context,
                    catalog=catalog,
                    session_id=session_id,
                    validation_errors=validation_errors,
                )
                _save_session(
                    Path(proj.project_dir),
                    session_id=session_id,
                    phase="plan",
                    round_idx=round_idx,
                )
                validation_errors = _validate_plan(plan, tree, cfg)
                if not validation_errors:
                    break
                log.warning(
                    "round %d: plan validation failed: %s",
                    round_idx,
                    validation_errors,
                )
            if validation_errors:
                raise RuntimeError(
                    "main-agent returned invalid ExperimentPlan after retries: "
                    + "; ".join(validation_errors)
                )
            assert plan is not None
            node_id = tree.add_planned_node(
                parent=plan.parent_node_id,
                round_idx=round_idx,
                plan=plan.model_dump(),
            )
            node = tree.get(node_id)
            tree.save()
            log.info(
                "round %d: + node %s parent=%s name=%s init_model=%s estimated_hours=%s",
                round_idx,
                node_id,
                node.get("parent"),
                node.get("experiment_name"),
                node.get("init_model_path"),
                node.get("estimated_hours"),
            )

        spec = _spec_from_node(node, eval_cfg)
        if not (study.plan and study.plan.get("experiments")):
            study.set_plan(
                {
                    "goal": spec.get("hypothesis") or f"round {round_idx}",
                    "node_id": node_id,
                    "parent_node_id": spec.get("parent_node_id"),
                    "init_model": spec.get("init_model"),
                    "experiments": [spec],
                }
            )
            log.info("round %d: wrote plan.json for node %s", round_idx, node_id)

        tree.mark_running(node_id)
        tree.save()
        log.info("round %d: executing node %s via sub-agent", round_idx, node_id)
        from dataos.agent import subagent_executor

        _publish_live(
            cfg,
            round_idx=round_idx,
            phase="executing",
            started=started,
            now=_clock(),
            node_id=node_id,
            node_name=node.get("experiment_name"),
            node_run_root=subagent_executor._run_root(cfg, study_id, spec["name"]),
            node_trains=bool((node.get("plan") or {}).get("train")),
        )

        deadline = started + cfg.budget.max_wall_clock_s if cfg.budget.max_wall_clock_s else None
        subagent_executor.fan_out(cfg=cfg, study_id=study_id, specs=[spec], deadline=deadline)

        exp_root = Path(cfg.workspace) / cfg.project_id / study_id / "experiments"
        score, mpath, artifacts, scripts, status, report_path = _read_report(
            exp_root,
            spec["name"],
        )
        tree.attach_report(
            node_id,
            report_path=report_path,
            status=status,
            score=score,
            model_path=mpath,
            artifacts=artifacts,
            scripts=scripts,
        )
        tree.save()
        log.info(
            "round %d: report node=%s status=%s score=%s model=%s",
            round_idx,
            node_id,
            status,
            score,
            mpath,
        )

        digest = study.summary()
        review_context = {
            "round_idx": round_idx,
            "node": tree.get(node_id),
            "study_digest": digest,
            "tree": tree.to_context(),
            "goal": cfg.goal or cfg.instruct,
        }
        log.info("round %d: asking main-agent for review", round_idx)
        _publish_live(
            cfg,
            round_idx=round_idx,
            phase="reviewing",
            started=started,
            now=_clock(),
            node_id=node_id,
            node_name=node.get("experiment_name"),
            last_score=score,
        )
        review, session_id = review_result(
            agent=main_agent,
            context=review_context,
            session_id=session_id,
        )
        _save_session(
            Path(proj.project_dir),
            session_id=session_id,
            phase="review",
            round_idx=round_idx,
        )
        review_payload = review.model_dump()
        study.review(conclusion=review_payload)
        tree.attach_review(node_id, review_path=str(study.review_path), review=review_payload)
        tree.save()
        log.info(
            "round %d: review interpretation=%r next=%r",
            round_idx,
            review.interpretation[:160],
            review.recommendation_for_future_growth[:160],
        )

        stats = tree.stats()
        best_node = tree.best_node()
        log.info(
            "round %d: tree best=%s n_nodes=%s n_growable=%s max_depth=%s",
            round_idx,
            (
                {
                    "id": best_node["id"],
                    "score": best_node["score"],
                    "depth": best_node["depth"],
                }
                if best_node
                else None
            ),
            stats["n_nodes"],
            stats["n_growable"],
            stats["max_depth"],
        )
        best = _better(best, digest.get("best"))
        events.append(
            {
                "round": round_idx,
                "study": study_id,
                "node": node_id,
                "parent": node.get("parent"),
                "status": status,
                "score": score,
                "best": digest.get("best"),
                "caution": digest.get("caution"),
            }
        )
        round_idx += 1
