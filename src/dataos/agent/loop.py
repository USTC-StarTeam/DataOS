"""Agentic autonomous loop.

The old main loop exposed the tree and deterministically picked three nodes per
round. The current loop is simpler and more agentic: Python maintains durable
state; one persistent leader dialog proposes the next single experiment plan;
Python validates/applies it; one experimenter executes data/train/eval; the main
dialog reviews the report; Python writes the tree.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from dataos.core.io import atomic_write_json
from dataos.agent import report as report_schema
from dataos.agent.eval_freeze import freeze_eval

from . import budget as _budget
from . import iterations as _iterations
from . import runtime
from . import runtime_compose
from . import roles
from .config import AutoRunConfig
from .schemas import ExperimentPlan

log = logging.getLogger(__name__)

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
    max_rounds = cfg.budget.max_rounds
    iterations_used = (
        round_idx + (1 if phase in {"executing", "reviewing"} else 0)
        if max_rounds is not None and round_idx >= 0
        else None
    )
    monitor.LIVE.publish(
        run_id=cfg.project_id,
        bench=cfg.target_bench,
        round=round_idx,
        phase=phase,
        remaining_hours=_remaining_hours(started, cfg, now),
        budget_hours=(cap / 3600.0) if cap else None,
        budget_unit="iterations" if max_rounds is not None else "wall_clock",
        budget_limit=max_rounds if max_rounds is not None else ((cap / 3600.0) if cap else None),
        budget_used=iterations_used,
        budget_remaining=(
            max(0, max_rounds - iterations_used)
            if max_rounds is not None and iterations_used is not None
            else None
        ),
        **extra,
    )
    monitor.write_live(Path(cfg.workspace), cfg.project_id)


def _safe_catalog(dataset_backend: str | None = None) -> list[dict]:
    try:
        import dataos

        rows = dataos.catalog()
        if dataset_backend is None:
            return rows
        return [row for row in rows if dataset_backend in row.get("dataset_backends", [])]
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
    """This report's score, via the schema — which single-sources it.

    The three-place fallback this replaces (``score`` → ``eval.macro_average`` → ``eval.score``)
    was not defensive, it was ambiguity: nothing said which of the three a report SHOULD write, so
    each place was authoritative somewhere and every consumer had to re-implement the precedence.
    """
    try:
        return report_schema.read(report).score
    except Exception as e:  # noqa: BLE001 — not report-shaped at all
        log.error("unreadable report: %s", e)
        return None


def _read_report(exp_root: Path, name: str):
    """Read one experiment report as score/eval_limit/model/scripts/status/path.

    ``eval_limit`` is how many items produced the score (``-1`` = the full benchmark). The leader
    must see it next to every score: a subset score and a full score are not comparable, and it is
    the leader that decides what to do about that.
    """
    from dataos.core.study import _slug

    path = exp_root / _slug(name) / "report.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None, None, None, {}, None, str(path)
    try:
        parsed = report_schema.read(raw)
    except Exception as e:  # noqa: BLE001 — a file that is not report-shaped at all
        # Loud on purpose. The version this replaces answered the same situation with a legacy
        # reader that produced `score=None, status="failed"` — indistinguishable from a genuine
        # failed experiment, and it cost 22 real scores before anyone looked.
        log.error("unreadable report at %s: %s", path, e)
        return None, None, None, {}, None, str(path)
    scripts = {
        stage: ref.model_dump()
        for stage in ("data", "train", "eval", "review")
        if (ref := getattr(getattr(parsed.stages, stage, None), "script", None)) is not None
    }
    return (
        parsed.score,
        parsed.stages.eval.limit if parsed.stages.eval else None,
        parsed.model_path,
        scripts,
        parsed.status,
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
            }
        )
    return rows


def _remaining_hours(started: float, cfg: AutoRunConfig, now: float) -> float | None:
    """What the leader is told it has left — off the SAME deadline the stop check and the
    experiment launcher use, so the three can never disagree. Reading it off an uncapped
    credit is how the leader kept planning experiments the launcher would refuse to run."""
    if cfg.budget.max_wall_clock_s is None:
        return None
    credit = runtime._queue_wait_s(cfg)
    deadline = _budget.credited_deadline(
        started_at=started, budget=cfg.budget, queue_credit_s=credit
    )
    return max(0.0, (deadline - now) / 3600.0)


def _effective_eval_limit(cfg: AutoRunConfig) -> int:
    if cfg.eval_limit:
        return cfg.eval_limit
    return 20 if cfg.ptb_profile == "smoke" else 200


def _full_eval_size(cfg: AutoRunConfig) -> int | None:
    """How many items the benchmark actually has, from the round-0 baseline's own eval.

    The baseline always runs at the full protocol (-1), so its ``n_samples`` is the only place the
    run learns the benchmark's real size — and that size is what turns a subset score into
    ``74.00@200/1319`` instead of a bare ``@200``. ``None`` until round 0 has landed.

    Read through the schema rather than off a raw key: the baseline report has TWO producers (the
    experimenter's contract and the investigator's recon prompt) and this reader was written
    against only the second one's spelling. Moving the field into the schema without moving this
    reader would have silently returned ``None`` here, dropping the ``/1319`` denominator — the one
    guard against a subset score being read as comparable (audit 15.4).
    """
    # read defensively: _validate_plan calls this, and tests validate against sparse doubles
    workspace, project_id = getattr(cfg, "workspace", None), getattr(cfg, "project_id", None)
    if not workspace or not project_id:
        return None
    path = (Path(workspace) / project_id / "round-0000" / "experiments" / "baseline"
            / "report.json")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    try:
        eval_stage = report_schema.read(report).stages.eval
    except Exception as e:  # noqa: BLE001 — not report-shaped at all
        log.error("unreadable baseline report at %s: %s", path, e)
        return None
    n = eval_stage.n_samples if eval_stage else None
    return int(n) if isinstance(n, (int, float)) and n > 0 else None


def _probe_floor(full: int | None) -> int | None:
    """Smallest subset worth spending an exploration probe on: a third of the benchmark.

    There used to be a ``max(..., 100)`` under this, justified as "below 100 items the noise
    (~50/sqrt(N)) exceeds any effect worth chasing". That floor, not the fraction, was what
    actually bound 5 of the 7 benches — it lifted arena/healthbench to ~40% and humaneval to
    61% of their sets, and for bfcl (100 items) and aime2025 (30) it equalled the full size,
    disabling subsetting outright. It bought no safety either: every pathology it was meant to
    stop (v4's ``eval_limit=1`` x16 and ``=32`` x23) is already caught by the fraction alone —
    gsm8k's floor is 439, healthbench's 81. And "a probe must have at least 100 items" is
    incoherent for a benchmark that only HAS 30. ``None`` = size unknown, no numeric guidance.

    With the constant gone, the old "if the floor would eat two thirds, use the full set"
    branch became dead code: a third is never at least two thirds.
    """
    if not full or full <= 0:
        return None
    return full // 3


_RECON_FILE = "recon.json"


def _load_recon(project_dir: Path) -> dict | None:
    """Read the run's recon.json (eval contract + baseline diagnosis + source profiles)."""
    try:
        data = json.loads((project_dir / _RECON_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _recon_context(recon: dict | None, *, recon_path: Path) -> dict | None:
    """The recon view for the leader's context bundle — EVERY round the same shape.

    What the leader can act on: the investigator's shortlist (already filtered against the eval
    contract), the ids that ``_validate_plan`` will accept, and the target shape its data recipe
    has to render toward. The annotated per-source profile table is never inlined: it runs to
    tens of thousands of characters, and inlining it once into a persistent dialog is the worst
    of both worlds — it occupies the history for the rest of the run and is still gone after a
    compaction. It stays on disk, and the leader (which has a shell) reads it when it wants to
    reach past the shortlist.

    ``parser`` is dropped from the contract: it names the scorer's function and file:line, which
    only whoever re-checks built targets against it can use — that is the experimenter."""
    if not recon:
        return recon
    contract = {k: v for k, v in (recon.get("eval_format_contract") or {}).items()
                if k != "parser"}
    profiles = [p for p in (recon.get("source_profiles") or []) if isinstance(p, dict)]
    return {
        "eval_format_contract": contract or None,
        "baseline_diagnosis": recon.get("baseline_diagnosis"),
        # the investigator's shortlist: small, already judged against the eval contract, and the
        # intended starting point for every plan
        "candidate_sources": recon.get("candidate_sources"),
        # every id _validate_plan accepts — the leader needs these to reach past the shortlist,
        # and nothing else about a source is needed to name it
        "legal_source_ids": [p.get("source") for p in profiles if p.get("source")],
        "recon_path": str(recon_path),
        "note": (
            "candidate_sources is the investigator's shortlist, filtered against the eval "
            "contract — plan over it. Any id in legal_source_ids is also accepted; the full "
            "annotated profiles (stats, real samples, per-source notes) are in recon_path, so "
            "read that file before reaching past the shortlist."
        ),
    }


def _plan_context(
    *,
    cfg: AutoRunConfig,
    tree,
    proj,
    round_idx: int,
    started: float,
    now: float,
    recon: dict | None = None,
) -> dict:
    full_size = _full_eval_size(cfg)  # read once: both the tree view and eval_policy need it
    train_frozen = runtime_compose.task_stage_config(cfg, "train").get("ownership") == "frozen"
    if cfg.budget.max_rounds is not None:
        budget_context = {
            "unit": "iterations",
            "iterations": cfg.budget.max_rounds,
            "iterations_used": round_idx,
            "iterations_remaining": max(0, cfg.budget.max_rounds - round_idx),
            "note": (
                "An iteration is consumed only after code activates an official suite target. "
                "Iteration 1 is the seeded random baseline."
            ),
        }
    else:
        budget_context = {
            "unit": "wall_clock",
            "max_wall_clock_s": cfg.budget.max_wall_clock_s,
            "remaining_hours": _remaining_hours(started, cfg, now),
            "note": (
                "remaining_hours is THIS invocation's wall-clock balance — the run "
                "continues across invocations (state persists in the workspace), so a low "
                "number means this process may stop soon, NOT that the run is over"
            ),
        }
    return {
        # only what a plan field can be chosen from: the goal, the bench, and the profile that
        # sizes it. Ids and config paths are host bookkeeping the leader cannot act on.
        "setting": {
            "goal": cfg.goal or cfg.instruct,
            "target_bench": cfg.target_bench,
            "ptb_profile": cfg.ptb_profile,
        },
        "budget": budget_context,
        "round_idx": round_idx,
        # RECON intel (gathered once before training): the shortlist, the legal ids and the eval
        # contract, inline every round; the annotated profile table stays on disk.
        # data.sources MUST be one of legal_source_ids — enforced in _validate_plan.
        "recon": _recon_context(recon, recon_path=Path(proj.project_dir) / _RECON_FILE),
        "tree": tree.to_context(full_size=full_size),
        "best": _global_best(proj),
        "recent_reviews": _recent_reviews(proj),
        "eval_policy": {
            "mode": "auto" if _effective_eval_limit(cfg) == -2 else "fixed_limit",
            "limit": _effective_eval_limit(cfg),
            # the benchmark's real item count (from the round-0 baseline) and the smallest
            # subset a probe may use; both None until the baseline has landed
            "full_size": full_size,
            "probe_floor": _probe_floor(full_size),
            # the RULE for using these lives in the leader's context prompt (one copy); this is
            # only the run's numbers, which the context cannot carry because they arrive with
            # round 0's baseline
        },
        # What CODE will reject, and nothing else: the leader's own role, the eval policy and the
        # data rules live in its context prompt, and repeating them here every round produced
        # three drifting copies of the same rule.
        "validated_by_code": [
            "parent_node_id must be one of tree.growable_node_ids.",
            "name must not repeat an existing experiment_name.",
            "data.sources must all appear in recon.legal_source_ids.",
            (
                "train must be empty; the task owns its frozen training workflow."
                if train_frozen
                else "train.model_name_or_path must be omitted; code injects the parent model path."
            ),
        ],
    }


def _validate_plan(plan, tree, cfg: AutoRunConfig, recon: dict | None = None) -> list[str]:
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
    if runtime_compose.task_stage_config(cfg, "train").get("ownership") == "frozen" \
            and (plan.train or {}):
        errors.append("train must be empty because this task's training workflow is frozen")
    if plan.estimated_hours is not None and plan.estimated_hours <= 0:
        errors.append("estimated_hours must be positive when provided")
    if _effective_eval_limit(cfg) == -2:
        el = plan.eval_limit
        if el is None or not isinstance(el, int) or isinstance(el, bool) or (el <= 0 and el != -1):
            errors.append(
                "eval_limit is required in auto eval mode: -1 (full benchmark) or a "
                "positive subset size — exploration probes may subset; a candidate-best "
                "confirmation must be -1 (full)"
            )
        else:
            # A probe below the floor cannot answer its own question: the noise (~50/sqrt(N))
            # exceeds any effect worth chasing, so it spends budget and returns nothing.
            floor = _probe_floor(_full_eval_size(cfg))
            if floor is not None and el != -1 and el < floor:
                errors.append(
                    f"eval_limit {el} is below the probe floor {floor} for this benchmark "
                    f"(a third of it): a subset that small is noise. "
                    f"Use -1 (full) or at least {floor}"
                )

    # Gate ②: every planned source must be one RECON profiled, and domain-relevant. The
    # DIRTY-format rejection was removed (2026-07-14, per user): off-format data may still teach
    # the model useful signal, and blocking it hard-crashed runs when the pool had no HIGH source.
    # RECON still profiles sources (raw stats + investigator notes) as INFO for the
    # leader — nothing here forbids any source. Skipped when recon has no profiles.
    profiles = {p.get("source"): p for p in (recon or {}).get("source_profiles", [])
                if isinstance(p, dict) and p.get("source")}
    if profiles:
        # Existence gate only: the source must be one recon profiled (no invented ids).
        # Relevance is the leader's judgment from the raw stats — the profiler no longer
        # emits quality labels, so there is nothing here to reject on.
        for src in (getattr(plan, "data", None) or {}).get("sources", []) or []:
            if src not in profiles:
                errors.append(
                    f"source {src!r} is not in recon.source_profiles — only use profiled pool "
                    f"sources"
                )
    return errors


def _spec_from_node(node: dict, eval_cfg: dict, cfg: AutoRunConfig) -> dict:
    plan = dict(node.get("plan") or {})
    train = dict(plan.get("train") or {})
    policy = runtime_compose.task_agent_config(cfg)
    init_model = (
        cfg.h100_base_model
        if policy.get("model_initialization") == "root"
        else node.get("init_model_path")
    )
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
        "eval_limit": plan.get("eval_limit"),  # the leader's pick in auto mode; None in fixed
        "hypothesis": plan.get("hypothesis", ""),
        "expected_signal": plan.get("expected_signal", ""),
        "subagent_brief": plan.get("subagent_brief", ""),
        "iteration": node.get("iteration"),
        "max_iterations": cfg.budget.max_rounds,
        "suite_dir": node.get("suite_dir"),
        "suite_target": node.get("suite_target"),
        "suite_target_state": node.get("suite_target_state"),
        "suite_run_dir": node.get("suite_run_dir"),
        "submission_path": node.get("submission_path"),
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
    """Run the agentic tree loop until the task-selected budget is spent."""

    import dataos

    from .backend import make_agent
    from .logging import setup_run_logging
    from .planner import next_plan
    from .reviewer import review_result
    from .state_tree import TreeState

    cfg.workspace = os.path.abspath(cfg.workspace)
    log, log_path = setup_run_logging(cfg.workspace, cfg.project_id)
    log.info(
        "run start: project=%s bench=%s backend=%s profile=%s budget=%s",
        cfg.project_id,
        cfg.target_bench,
        cfg.backend,
        cfg.ptb_profile,
        cfg.budget.as_dict(),
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
    data_stage = runtime_compose.task_stage_config(cfg, "data")
    task_agent = runtime_compose.task_agent_config(cfg)
    has_operator_toolbox = data_stage.get("toolbox") == "dataos-operator"
    catalog = _safe_catalog(data_stage.get("dataset_backend")) if has_operator_toolbox else []
    base_model = cfg.h100_base_model or cfg.base_model_id or "base"
    tree = TreeState.load(proj.project_dir, base_model)

    # persist the operator catalog once per run — later rounds just point at the file
    # (the leader has a shell; everything it needs must be queryable, not memorized)
    from .planner import _summarize_catalog

    catalog_path = Path(proj.project_dir) / "operator_catalog.json"
    atomic_write_json(catalog_path, _summarize_catalog(catalog))

    main_agent = make_agent("leader", cfg)
    session_id = _load_session(Path(proj.project_dir))
    if session_id:
        log.info("leader: resuming dialog session %s", session_id)

    started = _clock()
    runtime.set_invocation_wall_start(time.time())  # anchor for queue-credit overlap
    round_idx = len(_reviewed_studies(proj))
    best = _global_best(proj)
    events: list[dict] = []
    invalid_streak = 0  # consecutive rounds whose plans never validated (skip, don't crash)
    if round_idx:
        log.info(
            "resuming: %d reviewed node(s) on disk; continuing at round %d (best=%s)",
            round_idx,
            round_idx,
            best,
        )

    _start_monitor(cfg, log)

    # RECON gate ①: before ANY training, force a one-shot reconnaissance so the leader plans
    # with facts (eval format contract, base failure cause, per-source quality) instead of guessing.
    # Resume-safe: a valid recon.json on disk skips it. Skipped for the stub backend (no cluster).
    recon = _load_recon(Path(proj.project_dir))
    if recon is None and cfg.backend != "stub" and task_agent.get("investigator", True):
        log.info("RECON: mandatory reconnaissance before any experiment")
        _publish_live(cfg, round_idx=-1, phase="recon", started=started, now=_clock())
        recon_deadline = (
            started + cfg.budget.max_wall_clock_s if cfg.budget.max_wall_clock_s else None
        )
        try:
            recon = roles.run_investigator(
                cfg=cfg, proj=proj, catalog=catalog, deadline=recon_deadline
            )
        except Exception as e:  # noqa: BLE001 — recon failure must not brick the run
            log.error("RECON failed (%s); proceeding without source gating", e)
            recon = None

    while True:
        now = _clock()
        credit = 0.0 if cfg.backend == "stub" else runtime._queue_wait_s(cfg)
        reason = _budget.stop_reason(
            started_at=started,
            budget=cfg.budget,
            now=now,
            queue_credit_s=credit,
            rounds_completed=round_idx,
        )
        if reason:
            log.info("STOP: %s (after %d completed node(s))", reason, round_idx)
            if reason == "budget:iterations":
                suite_result = _iterations.finalize(cfg)
                if suite_result is not None:
                    events.append({"iteration_suite": suite_result})
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
            baseline = task_agent.get("baseline", "model_eval")
            if round_idx == 0 and baseline == "model_eval":
                # FIRST experiment is ALWAYS the baseline (evaluate the base model as-is,
                # no data, no training) — a code-fixed plan, not a leader decision: saves
                # a plan call and the leader can't open with a non-baseline experiment.
                log.info("round 0: baseline (code-fixed, no leader call)")
                plan = ExperimentPlan(
                    parent_node_id=tree.root_id,
                    name="baseline",
                    hypothesis="baseline: evaluate the base model as-is to anchor scoring",
                    estimated_hours=0.5,
                    data={},
                    train={},
                    expected_signal="reference score for every later experiment",
                    subagent_brief="BASELINE: evaluate the parent model directly; skip data and train.",
                )
            elif round_idx == 0 and isinstance(baseline, dict):
                log.info("round 0: task-configured baseline (no leader call)")
                plan = ExperimentPlan(
                    parent_node_id=tree.root_id,
                    name=str(baseline["name"]),
                    hypothesis=str(baseline["hypothesis"]),
                    estimated_hours=baseline.get("estimated_hours"),
                    data=dict(baseline.get("data") or {}),
                    train={},
                    expected_signal=str(baseline.get("expected_signal") or ""),
                    subagent_brief=str(baseline.get("subagent_brief") or ""),
                )
            for attempt in range(1, _PLAN_RETRIES + 1):
                if plan is not None:
                    break
                context = _plan_context(
                    cfg=cfg,
                    tree=tree,
                    proj=proj,
                    round_idx=round_idx,
                    started=started,
                    now=now,
                    recon=recon,
                )
                log.info("round %d: asking leader for plan (attempt %d)", round_idx, attempt)
                plan, session_id = next_plan(
                    agent=main_agent,
                    context=context,
                    catalog=catalog,
                    session_id=session_id,
                    validation_errors=validation_errors,
                    # the catalog is inlined on the first plan (rounds 0-1); afterwards the
                    # leader reads the persisted file on demand (validation failures re-inline
                    # it — a bad op name usually means the leader needs the list again)
                    catalog_pointer=(
                        None
                        if round_idx <= 1 or validation_errors
                        else str(catalog_path)
                    ),
                    catalog_path=str(catalog_path),
                    template=(task_agent.get("turns") or {}).get(
                        "leader_plan", "leader/turn/plan"
                    ),
                )
                _save_session(
                    Path(proj.project_dir),
                    session_id=session_id,
                    phase="plan",
                    round_idx=round_idx,
                )
                validation_errors = _validate_plan(plan, tree, cfg, recon)
                if not validation_errors:
                    break
                log.warning(
                    "round %d: plan validation failed: %s",
                    round_idx,
                    validation_errors,
                )
            if validation_errors:
                # never let a malformed plan kill the whole run: skip the round and
                # re-ask next round; stop cleanly only after a sustained invalid streak.
                invalid_streak += 1
                log.error(
                    "round %d: invalid plan after %d attempts (%s); skipping round "
                    "(streak %d)",
                    round_idx, _PLAN_RETRIES, "; ".join(validation_errors), invalid_streak,
                )
                if invalid_streak >= 3:
                    reason = f"leader_invalid_plans x{invalid_streak}"
                    log.error("STOP: %s", reason)
                    _publish_live(cfg, round_idx=round_idx, phase="done", started=started,
                                  now=_clock())
                    result = _finish(proj, reason, best, round_idx, events, tree)
                    log.info("final_model: %s", result.get("final_model"))
                    return result
                continue
            invalid_streak = 0
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

        iteration = _iterations.claim(cfg, node)
        if iteration is not None:
            node["iteration"] = iteration["iteration"]
            node["suite_dir"] = iteration["suite_dir"]
            node["suite_target"] = iteration["target_key"]
            node["suite_target_state"] = iteration["target_state"]
            node["suite_run_dir"] = iteration["run_dir"]
            node["submission_path"] = iteration["default_submission_path"]
            tree.save()
            log.info(
                "round %d: official iteration %d target=%s state=%s",
                round_idx,
                iteration["iteration"],
                iteration["target_key"],
                iteration["target_state"],
            )

        spec = _spec_from_node(node, eval_cfg, cfg)
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
        log.info("round %d: executing node %s via experimenter", round_idx, node_id)

        _publish_live(
            cfg,
            round_idx=round_idx,
            phase="executing",
            started=started,
            now=_clock(),
            node_id=node_id,
            node_name=node.get("experiment_name"),
            node_run_root=runtime._run_root(cfg, study_id, spec["name"]),
            node_trains=(
                runtime_compose.task_stage_config(cfg, "train").get("ownership") == "frozen"
                or bool((node.get("plan") or {}).get("train"))
            ),
        )

        # The SAME deadline the stop check above used — including this round's queue credit.
        # Computing it here without the credit is what produced the smol3 batch's 77 zombie
        # rounds (see budget.credited_deadline).
        deadline = (
            _budget.credited_deadline(started_at=started, budget=cfg.budget, queue_credit_s=credit)
            if cfg.budget.max_wall_clock_s
            else None
        )
        roles.fan_out(cfg=cfg, study_id=study_id, specs=[spec], deadline=deadline)

        exp_root = Path(cfg.workspace) / cfg.project_id / study_id / "experiments"
        score, eval_limit, mpath, scripts, status, report_path = _read_report(
            exp_root,
            spec["name"],
        )
        tree.attach_report(
            node_id,
            report_path=report_path,
            status=status,
            score=score,
            eval_limit=eval_limit,
            model_path=mpath,
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

        iteration_budget = (task_agent.get("budget") or {}).get("unit") == "iterations"
        if iteration_budget and (
            status != "completed" or score is None or not math.isfinite(score)
        ):
            reason = "execution:invalid_result"
            log.error("STOP: %s at node %s; inspect %s before resuming", reason, node_id, report_path)
            atomic_write_json(Path(proj.project_dir) / "execution_blocked.json", {
                "node_id": node_id, "round": round_idx, "report_path": report_path,
                "status": status, "score": score, "stop_reason": reason,
            })
            _publish_live(
                cfg, round_idx=round_idx, phase="blocked", started=started, now=_clock(),
                node_id=node_id, node_name=node.get("experiment_name"),
            )
            return _finish(proj, reason, best, round_idx, events, tree)

        if iteration_budget:
            (Path(proj.project_dir) / "execution_blocked.json").unlink(missing_ok=True)

        digest = study.summary()
        _node = tree.get(node_id)
        review_context = {
            "round_idx": round_idx,
            # what the leader predicted when it planned this node, lifted out of node.plan so the
            # review is asked about it instead of having to go looking
            "prediction": {
                "hypothesis": _node.get("hypothesis"),
                "expected_signal": (_node.get("plan") or {}).get("expected_signal"),
            },
            "node": _node,
            "study_digest": digest,
            "tree": tree.to_context(full_size=_full_eval_size(cfg)),
            "goal": cfg.goal or cfg.instruct,
        }
        log.info("round %d: asking leader for review", round_idx)
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
            }
        )
        round_idx += 1
