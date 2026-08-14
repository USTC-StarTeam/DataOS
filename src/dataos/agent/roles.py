"""Running one ROLE to a terminal result — the experimenter and the investigator.

Neither role is subordinate to the leader, and the module used to be called
``subagent_executor`` as if one were. The call graph says otherwise: ``loop.py`` is the only
orchestrator, and it invokes all three roles itself — the leader through
:func:`dataos.agent.backend.system_prompt`, and these two through :func:`fan_out` and
:func:`run_investigator`. Nothing in the leader's prompts can start an experiment; it proposes a
plan, ``_validate_plan`` checks it, and Python applies it. So the three are peers with different
jobs, and the old name encoded a hierarchy that no longer exists (and arguably never did).

The two roles run under separate lifecycles rather than a shared one, because what "finished"
means differs: an experimenter is done when its report.json is terminal (score first, review
second — see :func:`dataos.agent.report.is_terminal`), while the investigator is done when both
recon.json and the round-0 baseline report parse. Cluster and layout facts live in
:mod:`dataos.agent.runtime`; this module only knows how to drive an agent.

PTB allows only ONE H100 for the whole run, so experiments run SEQUENTIALLY (never in parallel)
under a shared single-GPU / 10h budget; ``report.json`` on disk is the source of truth.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from dataos.core.ids import _make_id
from dataos.core.io import atomic_write_json
from dataos.core.study import _slug

from dataos.agent import runtime
from dataos.agent import report as report_schema

from dataos import prompt as _prompt
from dataos.agent.backend import system_prompt
from dataos.agent.llm.config import build_agent

log = logging.getLogger(__name__)

# Let the role run FREE within its time budget. The real bound is the per-experiment
# DEADLINE (the timeout on each agent.run); if the experimenter stops before writing report.json
# the resume loop continues the SAME session. A high cap means max_turns ~never chops a live
# experiment into artificial 150-turn pieces (which only added latency + a failure surface).
ROLE_MAX_STEPS = 100_000
_EXP_TIMEOUT_SMOKE_S = 2 * 3600.0  # hard per-experiment timeout: 2 h (smoke)
_EXP_TIMEOUT_FORMAL_S = 10 * 3600.0  # PTB hard cap: 10 h / 1 H100
_STATE_FILE = "subagent_state.json"  # on-disk name kept: the resume path reads it in every
# archived run dir, so renaming it would strand old runs. The MODULE no longer claims a hierarchy.


def _exp_timeout(cfg, deadline: float | None) -> float:
    """Hard per-experiment timeout: a profile cap, never exceeding the remaining wall-clock."""
    cap = _EXP_TIMEOUT_SMOKE_S if cfg.ptb_profile == "smoke" else _EXP_TIMEOUT_FORMAL_S
    if deadline is not None:
        return max(60.0, min(cap, deadline - time.monotonic()))
    return cap


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _write_state(path: Path, state: dict) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_write_json(path, state)


def _report_from_disk(path: Path) -> dict | None:
    data = _read_json(path)
    return data if isinstance(data, dict) else None


def _sweep_point(eval_dir: str, summary: dict) -> dict:
    """One eval dir's numbers, in the report's own units.

    ``summary.json`` carries the metric twice: ``score`` already scaled to a percentage, and
    ``metrics`` in the harness's native fraction. Copying stderr straight out of ``metrics`` next
    to a percentage score is how the same report ended up saying ``score=0.89, stderr=0.0044``
    while its siblings said ``30.36 / 2.13`` — the same quantity, two scales, one block.
    """
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    metric_key = summary.get("metric_key")
    stderr = None
    if isinstance(metric_key, str):
        raw = metrics.get(metric_key.replace(",", "_stderr,", 1))
        stderr = float(raw) * 100 if isinstance(raw, (int, float)) else None
    return {
        "eval_dir": eval_dir,
        "score": _numeric_score(summary),
        "limit": summary.get("limit") if isinstance(summary.get("limit"), int) else None,
        "n_samples": summary.get("n_samples") if isinstance(summary.get("n_samples"), int) else None,
        "stderr": stderr,
        "elapsed_s": (summary.get("elapsed_seconds")
                      if isinstance(summary.get("elapsed_seconds"), (int, float)) else None),
    }


def _stamp_measurements(*, cfg, study_id: str, name: str, report: dict) -> None:
    """Replace the eval block's numbers with the frozen harness's own, in place.

    The experimenter restating ``score`` / ``limit`` / ``n_samples`` / ``stderr`` from
    ``summary.json`` makes it a courier of numbers that already exist in a machine-written file,
    and a courier is the wrong thing to validate — so the runtime reads the files instead. Two
    things fall out of doing it here:

    * the whole sweep becomes visible. 15 of the 29 nodes of the 2026-08-13 batch scored between
      two and six checkpoints on the FULL benchmark and reported only the maximum; a champion's
      30.36 came from checkpoint 12 while its own final weights scored 28.12, and nothing
      downstream could see it. The eval dirs were on disk the whole time.
    * the declared score gets checked. Picking the eval dir whose score MATCHES what the
      experimenter claimed both identifies the winning checkpoint and turns "claimed a number no
      eval produced" from an invisible event into a logged one.
    """
    try:
        summaries = runtime._remote_eval_summaries(cfg, study_id, name)
    except Exception as e:  # noqa: BLE001 — the exec host is an I/O boundary, not a contract
        log.warning("experimenter[%s]: cannot read eval summaries (%s); "
                    "leaving the experimenter's own numbers in place", name, e)
        return
    points = [_sweep_point(d, s) for d, s in summaries if _numeric_score(s) is not None]
    if not points:
        return
    stages = report.setdefault("stages", {})
    if not isinstance(stages, dict):
        return
    eval_stage = stages.setdefault("eval", {})
    if not isinstance(eval_stage, dict):
        return

    claimed = eval_stage.get("score")
    winner = next(
        (p for p in points
         if isinstance(claimed, (int, float)) and abs(p["score"] - claimed) < 1e-6),
        None,
    )
    if winner is None:
        winner = points[0]  # newest
        if isinstance(claimed, (int, float)):
            log.warning(
                "experimenter[%s]: reported score=%s matches none of the %d scored eval dirs %s "
                "— stamping the newest (%s) instead",
                name, claimed, len(points), [p["score"] for p in points], winner["score"],
            )
    eval_stage.update({k: v for k, v in winner.items() if k != "eval_dir" and v is not None})
    eval_stage["summary_dir"] = winner["eval_dir"]
    if len(points) > 1:
        eval_stage["checkpoint_sweep"] = points
        log.info("experimenter[%s]: %d scored eval dirs, winner %s @ %s",
                 name, len(points), winner["score"], winner["eval_dir"])


def _stamp_report(*, cfg, study_id: str, spec: dict, exp_dir: Path, eval_limit: int,
                  report: dict) -> dict:
    """Attach runtime-owned provenance to any report the experimenter wrote.

    Deliberately thin. It used to stamp seven guessed paths under ``artifacts`` (19.3% of every
    report's bytes, and the leader opens none of them) and to seed ``scripts`` with empty strings,
    which is how the previous batch reported 17 scripts as ``""`` — a key whose value the runtime
    invents is not provenance. Paths live in ``state.json`` now; a script the experimenter actually
    wrote is declared per stage, with a hash.

    Two things are worth stamping, and they are worth it for opposite reasons. The model path
    because the tree continues from it and an experimenter that trained into the standard location
    often does not bother repeating it. The eval measurements because they are not the
    experimenter's to report at all — see :func:`_stamp_measurements`.
    """
    name = spec["name"]
    report.setdefault("id", _make_id("exp"))
    stages = report.setdefault("stages", {})
    if isinstance(stages, dict):
        train = stages.get("train")
        if isinstance(train, dict) and not train.get("model_path"):
            train["model_path"] = runtime._paths(cfg, study_id, name, exp_dir)["model"]
    _stamp_measurements(cfg=cfg, study_id=study_id, name=name, report=report)
    return report


def _failed_report(*, error: str) -> dict:
    # `limit: 0` is the honest value here: nothing was scored. Writing the requested eval_limit
    # would claim a caliber this node never had. The caller stamps.
    return {
        "id": _make_id("exp"),
        "stages": {"eval": {"status": "failed", "score": None, "limit": 0, "error": error}},
    }


def _numeric_score(data: dict) -> float | None:
    score = data.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _score_of(partial: dict) -> float | None:
    """The partial report's score, or None if it cannot be read at all.

    Salvage is the LAST-RESORT path — it exists because something already went wrong — so the
    one thing it must not do is add a second failure. It did: reading the partial report
    unguarded, it hit a leaf whose type the experimenter had filled with prose and raised
    ValidationError out of ``fan_out``, out of ``run_auto``, out of the process. The run died
    with 17 minutes of budget left, on a node that had a score, in the function whose entire
    purpose is to recover a score from a report that is not clean.

    An unreadable partial simply has no score here; the remote ``summary.json`` sweep below is
    the other source and is untouched by whatever the experimenter wrote.
    """
    try:
        return report_schema.read(partial).score
    except Exception as e:  # noqa: BLE001 — any unreadable partial means "no score here"
        log.warning("salvage: partial report.json is unreadable (%s); trying the eval dirs", e)
        return None


def _salvage_report(*, cfg, study_id: str, spec: dict, exp_dir: Path, eval_limit: int,
                    partial: dict | None, error: str) -> dict | None:
    """Rescue a COMPLETED score when the experimenter died after the eval produced one
    (CLI stall / budget cut mid-analysis) but before writing a terminal report.

    Two sources, in order: the partial local report.json (the contract's incremental
    score-first write), then the deterministic remote {run_root}/eval/.../summary.json.
    Returns None when no real score exists anywhere (caller falls back to failed)."""
    source = None
    report = None
    if partial is not None and _score_of(partial) is not None:
        report, source = dict(partial), "partial report.json"
    else:
        for eval_dir, summary in runtime._remote_eval_summaries(cfg, study_id, spec["name"]):
            if _numeric_score(summary) is not None:
                report, source = {
                    "stages": {
                        "eval": {
                            "status": "completed",
                            "score": _numeric_score(summary),
                            "limit": summary.get("limit")
                            if isinstance(summary.get("limit"), int)
                            else eval_limit,
                        }
                    },
                }, f"remote summary.json ({eval_dir})"
                break
    if report is None:
        return None
    # The score IS real; the cut was operational — so the EVAL stage completed even though the
    # experiment did not. Salvage cannot invent the review that was never written, and a report
    # with no review would read as mid-run to the resume check, so say plainly why it stops here.
    stages = report.setdefault("stages", {})
    if isinstance(stages, dict):
        eval_stage = stages.setdefault("eval", {})
        if isinstance(eval_stage, dict):
            eval_stage["status"] = "completed"
        stages["review"] = {
            "status": "skipped",
            "failure_classes": {},
            "sampled": False,
            "anomalies": "unknown",
            "conclusion": f"no review: the experimenter was cut off after the eval scored ({error})",
        }
    report["salvaged"] = True
    report["error"] = f"salvaged from {source}; the experimenter was cut off: {error}"
    log.warning("experimenter[%s]: salvaged score=%s from %s",
                spec["name"], report_schema.read(report).score, source)
    return report


def _state_seed(*, cfg, study_id: str, spec: dict, exp_dir: Path, report_path: Path,
                eval_limit: int) -> dict:
    name = spec["name"]
    return {
        "name": name,
        "study_id": study_id,
        "status": "running",
        "session_id": None,
        "report_path": str(report_path),
        "state_path": str(exp_dir / _STATE_FILE),
        "artifacts": runtime._paths(cfg, study_id, name, exp_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _runtime_prompt_block(*, cfg, study_id: str, spec: dict, exp_dir: Path,
                          eval_limit: int) -> dict:
    runtime_config = runtime._runtime_config(cfg)
    runtime_config.setdefault("paths", {})["run_root"] = runtime._run_root(cfg, study_id, spec["name"])
    runtime_config["paths"]["local_report"] = str(exp_dir / "report.json")
    runtime_config["paths"]["local_state"] = str(exp_dir / _STATE_FILE)
    parent_model = (
        (spec.get("train") or {}).get("model_name_or_path")
        or spec.get("init_model")
        or cfg.h100_base_model
    )
    runtime_config["model"] = {
        "base_model": parent_model,
        "parent_model": parent_model,
        "root_base_model": cfg.h100_base_model,
        "base_model_id": getattr(cfg, "base_model_id", None),
    }
    runtime_config["posttrainbench"] = {
        "bench": cfg.target_bench,
        "profile": cfg.ptb_profile,
        "eval_limit": eval_limit,
    }
    if getattr(cfg, "runtime_config_path", None):
        runtime_config["source"] = str(cfg.runtime_config_path)
    return runtime_config


def _resolved_sources(cfg, sources: list) -> list[dict]:
    """The planned source ids, resolved to what an experimenter has to know before it can read one.

    The contract has told experimenters "the task gives you ``resolved_sources`` (exact on-disk
    paths)" for as long as it has existed, and the task never gave them anything of the kind — a
    promised input that is not there is worse than no promise, because it stops nobody from
    guessing. Four of four runs of the 2026-08-13 batch guessed, in four different spellings
    (``POOL / name``, ``os.path.join(pool, name)``, ...), and then crashed twice more on the shape:

        KeyError: "Invalid key: 0. Please first select a split. For example:
                   `my_dataset_dictionary['test'][0]`."   <- mmlu_aux, whose splits are
        AttributeError: 'DatasetDict' object has no attribute 'select'/'features'

    Every one of the pool's 106 sources is a ``DatasetDict``, so this is not a source-dependent
    hazard, it is certain. And "just take ['train']" is not a rule that can be written down: 47
    sources carry more than one split and 5 carry no ``train`` at all (agentinstruct's are
    os/db/alfworld/webshop/kg/mind2web; ultrafeedback's are train_prefs/train_sft/...). Split names
    are per-source DATA. Data has to be handed over, not documented.

    Nothing here is newly discovered: recon profiled all 106 sources with their split names and row
    counts, and ``_validate_plan`` already reads that file to check the plan's source ids against
    it. The profile simply never reached the experimenter that had to open the files.
    """
    profiles = _read_json(Path(cfg.workspace) / cfg.project_id / _RECON_FILE) or {}
    by_id = {
        p.get("source"): p
        for p in (profiles.get("source_profiles") or [])
        if isinstance(p, dict) and p.get("source")
    }
    pool = (runtime._runtime_config(cfg).get("paths") or {}).get("fixed_pool") or ""
    resolved = []
    for sid in sources:
        profile = by_id.get(sid) or {}
        resolved.append(
            {
                "id": sid,
                "path": f"{pool.rstrip('/')}/{sid}" if pool else None,
                # every source is a DatasetDict; this is the only place the split names exist
                "splits": profile.get("splits"),
                "shape": profile.get("shape"),
                "n_rows": profile.get("n_rows"),
            }
        )
    return resolved


def _task_prompt(*, cfg, study_id: str, spec: dict, exp_dir: Path, eval_limit: int,
                 remaining_minutes: int | None = None) -> str:
    bench = cfg.target_bench
    run_root = runtime._run_root(cfg, study_id, spec["name"])
    parent_model = (
        (spec.get("train") or {}).get("model_name_or_path")
        or spec.get("init_model")
        or cfg.h100_base_model
    )
    data = spec.get("data") or {}
    recipe = {"sources": data.get("sources", []), "recipe": data.get("recipe", [])}
    runtime_block = _runtime_prompt_block(
        cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir, eval_limit=eval_limit,
    )
    from . import runtime_compose, stages as _stages, train_spec as _train_spec
    env_ops, task_def, resolved = runtime_compose.render_sections(
        cfg, parent_model=parent_model, eval_limit=eval_limit,
    )
    stage_blocks = _stages.render_stage_blocks(_stages.build_stages(resolved))
    train = spec.get("train") or {}
    train_frozen = bool(resolved.get("task", {}).get("train"))  # task ships a train/ dir
    train_spec_json = ""
    if train.get("finetuning_type") and not train_frozen:  # our method; skip if frozen or baseline
        ts = _train_spec.build_train_spec(
            resolved, name=spec["name"], plan_train=train, parent_model=parent_model,
            run_root=run_root, time_h=2.0 if cfg.ptb_profile == "smoke" else 10.0,
        )
        train_spec_json = json.dumps(ts, ensure_ascii=False, indent=2)
    return _prompt.render(
        "experimenter/turn/experiment",
        exp_name=spec["name"],
        env_ops=env_ops,
        task_def=task_def,
        stage_blocks=stage_blocks,
        train_spec_json=train_spec_json,
        bench=bench,
        profile=cfg.ptb_profile,
        parent_model=parent_model,
        root_base_model=cfg.h100_base_model,
        node_id=spec.get("node_id"),
        parent_node_id=spec.get("parent_node_id"),
        eval_limit=eval_limit,
        is_smoke=cfg.ptb_profile == "smoke",
        recipe_json=json.dumps(recipe, ensure_ascii=False),
        resolved_sources_json=json.dumps(
            _resolved_sources(cfg, data.get("sources") or []), ensure_ascii=False, indent=1
        ),
        train_json=json.dumps(spec.get("train") or {}, ensure_ascii=False),
        subagent_brief=str(spec.get("subagent_brief") or "(none)"),
        runtime_config_json=json.dumps(runtime_block, ensure_ascii=False, indent=2, sort_keys=True),
        run_root=run_root,
        report_path=str(exp_dir / "report.json"),
        state_path=str(exp_dir / _STATE_FILE),
        remaining_minutes=remaining_minutes,
    )


def _continue_prompt(state: dict, last_text: str, remaining_minutes: int | None = None) -> str:
    return _prompt.render(
        "experimenter/turn/experiment_continue",
        report_path=state["report_path"],
        state_path=state["state_path"],
        state_json=json.dumps(state, ensure_ascii=False),
        last_text=last_text[-2000:] if last_text else "(no previous assistant text captured)",
        remaining_minutes=remaining_minutes,
    )


def _sleep_before_resume(cfg, remaining: float) -> None:
    if remaining <= 0:
        return
    delay = max(float(getattr(cfg, "poll_interval_s", 10.0)), 30.0)
    time.sleep(min(delay, remaining))


def _run_one(cfg, study_id: str, spec: dict, exp_root: Path, timeout: float | None = None) -> tuple[str, dict]:
    name = spec["name"]
    exp_dir = exp_root / _slug(str(name))
    exp_dir.mkdir(parents=True, exist_ok=True)
    report_path = exp_dir / "report.json"
    state_path = exp_dir / _STATE_FILE
    eval_limit = spec.get("eval_limit") or getattr(cfg, "eval_limit", None)
    if eval_limit is None:
        eval_limit = 20 if cfg.ptb_profile == "smoke" else 200
    existing_report = _report_from_disk(report_path)
    if existing_report is not None and report_schema.is_terminal(existing_report):
        report = _stamp_report(
            cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
            eval_limit=eval_limit, report=existing_report,
        )
        atomic_write_json(report_path, report)
        _parsed = report_schema.read(report)
        log.info("experimenter[%s]: existing terminal report status=%s score=%s",
                 name, _parsed.status, _parsed.score)
        return name, report

    agent = build_agent(
        cfg,
        role="experimenter",  # cwd comes from role_cwd (outside the repo); the whole
        instructions=system_prompt("experimenter", cfg),  # toolchain is ssh/scp + abs paths
        max_steps=max(cfg.max_steps, ROLE_MAX_STEPS),
        transcript=str(exp_dir / "agent.jsonl"),  # per-experiment run record (monitor + archive)
    )
    # The model this NODE starts from, resolved the same way the prompt resolves it — not
    # `cfg.h100_base_model`, which is the run's base and is only the node's parent for a
    # root node. A continuation node logged "base=<run base>" two lines under its own
    # "init_model=<checkpoint-400>", which reads as the continuation having been dropped.
    parent = (
        (spec.get("train") or {}).get("model_name_or_path")
        or spec.get("init_model")
        or cfg.h100_base_model
    )
    log.info("experimenter[%s]: starting (bench=%s parent=%s timeout=%ss)",
             name, cfg.target_bench, parent, int(timeout) if timeout else None)
    state = _read_json(state_path) or _state_seed(
        cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
        report_path=report_path, eval_limit=eval_limit,
    )
    state.setdefault("artifacts", runtime._paths(cfg, study_id, name, exp_dir))
    state.setdefault("report_path", str(report_path))
    state.setdefault("state_path", str(state_path))
    _write_state(state_path, state)

    started = time.monotonic()
    deadline = started + timeout if timeout else None
    prompt = _task_prompt(cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
                          eval_limit=eval_limit,
                          remaining_minutes=int(timeout / 60) if timeout else None)
    last_text = ""
    while True:
        remaining = None if deadline is None else max(
            0.0, deadline + runtime._queue_wait_s(cfg) - time.monotonic()
        )
        if remaining is not None and remaining <= 0:
            break
        try:
            res = agent.run(prompt, resume=state.get("session_id"), timeout=remaining)
        except Exception as e:  # noqa: BLE001 — transient transport drops are already retried
            # (reconnect+resume) inside dataos_agent via CliAgent.run; reaching here means a genuine
            # blocker or the wall-clock/PTB budget is spent, so record it and stop.
            state["status"] = "blocked"
            state["last_error"] = str(e)
            _write_state(state_path, state)
            log.error("experimenter[%s]: agent error: %s", name, e)
            break

        last_text = (getattr(res, "text", "") or "").strip()
        sid = getattr(res, "session_id", None)
        if sid:
            state["session_id"] = sid
        state["status"] = "running"
        state["last_message"] = last_text[-4000:]
        _write_state(state_path, state)
        log.info("experimenter[%s]: message (%d chars, session=%s): %s", name, len(last_text),
                 state.get("session_id"), last_text[:800] + ("…" if len(last_text) > 800 else ""))

        report = _report_from_disk(report_path)
        if report is not None and report_schema.is_terminal(report):
            state["status"] = report_schema.read(report).status
            _write_state(state_path, state)
            break

        remaining = None if deadline is None else max(
            0.0, deadline + runtime._queue_wait_s(cfg) - time.monotonic()
        )
        if remaining is not None and remaining <= 0:
            break
        log.info("experimenter[%s]: no report.json yet; resuming same session", name)
        _sleep_before_resume(cfg, remaining if remaining is not None else 30.0)
        prompt = _continue_prompt(
            state, last_text,
            remaining_minutes=int(remaining / 60) if remaining is not None else None,
        )

    report = _report_from_disk(report_path)
    if report is None or not report_schema.is_terminal(report):
        err = "experimenter produced no terminal report.json before the PTB/experiment deadline"
        if state.get("last_error"):
            err = f"{err}: {state['last_error']}"
        # the eval may have finished before the cut — rescue a real score if one exists
        salvaged = _salvage_report(
            cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
            eval_limit=eval_limit, partial=report, error=err,
        )
        report = salvaged if salvaged is not None else _failed_report(error=err)
    report = _stamp_report(
        cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
        eval_limit=eval_limit, report=report,
    )
    atomic_write_json(report_path, report)
    _parsed = report_schema.read(report)
    log.info("experimenter[%s]: status=%s score=%s run_root=%s",
             name, _parsed.status, _parsed.score, runtime._run_root(cfg, study_id, name))
    return name, report


_RECON_FILE = "recon.json"
# No separate clock on recon: the whole run plans over its output, so a thin recon costs far more
# than the minutes it takes to finish. Only the run's own wall-clock ``deadline`` bounds it.


def run_investigator(*, cfg, proj, catalog=None, deadline: float | None = None) -> dict:
    """RECON stage — a one-shot experimenter that gathers ground-truth intel BEFORE any training
    and writes ``recon.json`` (eval_format_contract + baseline_diagnosis + candidate_sources +
    source_profiles) AND runs the round-0 BASELINE eval itself (frozen protocol on the base
    model, report written to the round-0000 path — the loop's existing-report short-circuit
    then skips the baseline experimenter; if it's missing, round 0 runs as before).
    Resume-safe: if a valid recon.json already exists on disk, the loop skips calling this."""
    del catalog  # recon needs the pool + eval code, not the operator catalog
    recon_path = Path(proj.project_dir) / _RECON_FILE
    runtime_config = runtime._runtime_config(cfg)
    paths = runtime_config.get("paths", {})
    cluster = runtime_config.get("cluster", {})
    deploy_root = paths.get("deploy_root", "")
    eval_env = paths.get("eval_env", "")
    task_dir = runtime._task_assets_dir(runtime_config)
    eval_limit = getattr(cfg, "eval_limit", None)
    if eval_limit is None:
        eval_limit = 20 if cfg.ptb_profile == "smoke" else -1
    if eval_limit == -2:
        eval_limit = -1  # AUTO governs experiments; the baseline reference is always full
    limit_tag = str(eval_limit)
    baseline_run_root = f"{paths.get('runs_base', runtime._RUNS_BASE)}/{cfg.project_id}/round-0000/baseline"
    baseline_report = (
        Path(proj.project_dir) / "round-0000" / "experiments" / "baseline" / "report.json"
    )
    from . import runtime_compose

    env_ops, task_def, _ = runtime_compose.render_sections(
        cfg, parent_model=cfg.h100_base_model, eval_limit=eval_limit,
    )
    prompt = _prompt.render(
        "investigator/turn/recon",
        target_bench=cfg.target_bench,
        base_model=cfg.h100_base_model,
        fixed_pool=paths.get("fixed_pool", ""),
        deploy_root=deploy_root,
        eval_env_python=f"{eval_env}/bin/python" if eval_env else "python",
        cluster_json=json.dumps(
            {k: cluster.get(k) for k in ("host", "user", "port", "key")}, ensure_ascii=False
        ),
        profile_script=f"{deploy_root}/resources/recon/{runtime_config.get('task', 'posttrainbench')}/profile_source.py",
        lm_bridge=f"{task_dir}/eval",
        recon_path=str(recon_path),
        baseline_run_root=baseline_run_root,
        baseline_eval_dir=f"{baseline_run_root}/eval/{cfg.target_bench}_limit{limit_tag}",
        baseline_report_path=str(baseline_report),
        eval_limit=eval_limit,
        env_ops=env_ops,
        task_def=task_def,
    )
    agent = build_agent(
        cfg,
        role="investigator",
        instructions=system_prompt("investigator", cfg),
        max_steps=max(cfg.max_steps, ROLE_MAX_STEPS),
        transcript=str(Path(proj.project_dir) / "investigator.jsonl"),
    )
    # bounded only by the run's own budget; no recon-specific cap
    cap = float("inf") if deadline is None else max(60.0, deadline - time.monotonic())
    log.info("recon: starting reconnaissance (bench=%s, cap=%ss)", cfg.target_bench, int(cap))
    started = time.monotonic()
    session_id = None
    last_text = ""
    while time.monotonic() - started < cap:
        remaining = None if cap == float("inf") else cap - (time.monotonic() - started)
        try:
            res = agent.run(prompt, resume=session_id, timeout=remaining)
        except Exception as e:
            # A dead investigator means the run has no pool intel and no baseline. Proceeding
            # would hand the leader a hollow recon it cannot tell from a real one — fail loudly.
            raise RuntimeError(f"recon: investigator failed: {e}") from e
        session_id = getattr(res, "session_id", None) or session_id
        last_text = (getattr(res, "text", "") or "").strip()
        recon = _read_json(recon_path)
        baseline_done = _read_json(baseline_report) or {}
        if isinstance(recon, dict) and recon.get("source_profiles") \
                and baseline_done.get("status") in {"completed", "failed"}:
            log.info("recon: done — %d source profiles, base_cause=%s, baseline=%s",
                     len(recon.get("source_profiles") or []),
                     (recon.get("baseline_diagnosis") or {}).get("cause"),
                     baseline_done.get("score"))
            return recon
        prompt = (
            "recon.json is not yet complete at %s, or the baseline report is missing at %s. "
            "Finish BOTH deliverables: recon.json (eval_format_contract, baseline_diagnosis, "
            "candidate_sources, source_profiles) and the baseline report.json."
            % (recon_path, baseline_report)
        )
    # The run's wall-clock budget ran out first. NEVER synthesise a stand-in recon: an empty
    # shortlist and empty profiles are indistinguishable from a real survey once the leader has
    # them, so it would plan blind for the rest of the run and nothing would say why.
    raise RuntimeError(
        f"recon: incomplete when the run budget ran out — recon.json at {recon_path} and/or the "
        f"baseline report at {baseline_report} never became complete. Last agent message: "
        f"{last_text[:300]}"
    )


def fan_out(*, cfg, study_id: str, specs: list, deadline: float | None = None) -> dict[str, dict]:
    """Run each spec as an agentic experimenter. PTB allows only ONE H100 for the whole run, so
    experiments execute SEQUENTIALLY (never in parallel), sharing the single-GPU / 10h budget.
    ``deadline`` (monotonic) stops launching further experiments once the budget is spent."""
    if not cfg.target_bench:
        raise ValueError("the role runner needs cfg.target_bench (the single PTB bench)")
    if not cfg.h100_base_model:
        raise ValueError("the role runner needs cfg.h100_base_model (the -Base model path)")
    missing_runtime = runtime._missing_runtime_config_keys(cfg)
    if missing_runtime:
        raise ValueError(
            "the role runner needs runtime config with: " + ", ".join(missing_runtime)
        )
    exp_root = Path(cfg.workspace) / cfg.project_id / study_id / "experiments"
    log.info("fan_out: %d agentic experimenter(s), SEQUENTIAL (PTB: 1 H100 at a time) bench=%s",
             len(specs), cfg.target_bench)
    reports: dict[str, dict] = {}
    for spec in specs:
        name = spec["name"]
        if deadline is not None and time.monotonic() >= deadline:
            log.info("fan_out: wall-clock budget spent — skipping remaining experiment %s", name)
            reports[name] = {
                "id": _make_id("exp"), "name": name, "status": "failed", "score": None,
                "error": "skipped: wall-clock budget exhausted",
                "stages": {"train": {"status": "failed"}, "eval": {"status": "failed"}},
            }
            continue
        _, report = _run_one(cfg, study_id, spec, exp_root, timeout=_exp_timeout(cfg, deadline))
        reports[name] = report
    return reports
