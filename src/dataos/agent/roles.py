"""Running one ROLE to a terminal result — the experimenter and the investigator.

Neither role is subordinate to the leader, and the module used to be called
``subagent_executor`` as if one were. The call graph says otherwise: ``loop.py`` is the only
orchestrator, and it invokes all three roles itself — the leader through
:func:`dataos.agent.backend.system_prompt`, and these two through :func:`fan_out` and
:func:`run_investigator`. Nothing in the leader's prompts can start an experiment; it proposes a
plan, ``_validate_plan`` checks it, and Python applies it. So the three are peers with different
jobs, and the old name encoded a hierarchy that no longer exists (and arguably never did).

The two roles run under separate lifecycles rather than a shared one, because what "finished"
means differs: an experimenter is done when its report.json is terminal, while the investigator is
done when recon.json has its source profiles AND the round-0 baseline report is terminal. Both
lifecycles ask :func:`dataos.agent.report.is_terminal` — one function, so the two cannot drift
apart, which they had. Cluster and layout facts live in :mod:`dataos.agent.runtime`; this module
only knows how to drive an agent.

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
_STATE_FILE = "subagent_state.json"  # on-disk name kept: the resume path reads it in every
# archived run dir, so renaming it would strand old runs. The MODULE no longer claims a hierarchy.

_EXEC_HOST_NETWORK_ERRORS = (
    "no route to host",
    "network is unreachable",
    "connection timed out",
    "connection refused",
    "connection reset by peer",
    "connection closed by remote host",
    "could not resolve hostname",
    "kex_exchange_identification",
)


def _exp_timeout(cfg, deadline: float | None) -> float | None:
    """Wall-clock tasks have a hard cap; iteration-budget tasks run to a terminal result."""
    if cfg.budget.max_wall_clock_s is None:
        return None
    cap = cfg.budget.max_wall_clock_s
    if cfg.ptb_profile == "smoke":
        cap = min(cap, _EXP_TIMEOUT_SMOKE_S)
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


def _execution_host_wait(report: dict | None) -> tuple[str, str] | None:
    """Return the in-flight stage whose durable handle is hidden by a network failure.

    Losing the route to the execution host says nothing about the remote job's terminal state.
    A model can nevertheless mistake the failed *poll* for a failed *stage* and write a terminal
    report.  Accepting that report spends an iteration and abandons the still-owned handle.  Keep
    this whitelist narrow: a stage without a handle, or an actual tool/job error, remains failed.
    """
    if not isinstance(report, dict):
        return None
    status = report.get("status")
    stages = report.get("stages")
    if not isinstance(stages, dict):
        return None
    for name, stage in stages.items():
        if status not in {"failed", name} or not isinstance(stage, dict):
            continue
        error = stage.get("error")
        if not isinstance(error, str) or not error:
            continue
        if not (stage.get("handle") or stage.get("job_id")):
            continue
        if any(marker in error.lower() for marker in _EXEC_HOST_NETWORK_ERRORS):
            return str(name), error
    return None


def _reopen_execution_host_wait(report_path: Path, report: dict) -> tuple[str, str] | None:
    pending = _execution_host_wait(report)
    if pending is None:
        return None
    stage, _ = pending
    if report.get("status") == "failed":
        report["status"] = stage
        atomic_write_json(report_path, report)
    return pending


def _wait_for_execution_host(cfg, *, deadline: float | None) -> bool:
    """Wait without spending agent turns until the configured execution host is reachable."""
    attempt = 0
    while True:
        attempt += 1
        try:
            runtime.exec_host_command(cfg, ["true"], timeout=10)
            return True
        except Exception as exc:  # noqa: BLE001 - classify a narrow transport whitelist below
            if not any(marker in str(exc).lower() for marker in _EXEC_HOST_NETWORK_ERRORS):
                raise
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            if attempt == 1 or attempt % 10 == 0:
                log.warning("execution host unreachable; preserving stage handle and waiting: %s", exc)
            delay = max(float(getattr(cfg, "poll_interval_s", 10.0)), 60.0)
            time.sleep(delay if remaining is None else min(delay, remaining))


def _record_execution_host_recovery(report_path: Path, report: dict, stage: str, error: str) -> None:
    stage_data = (report.get("stages") or {}).get(stage)
    if not isinstance(stage_data, dict):
        return
    if stage_data.get("error") == error:
        stage_data.pop("error", None)
    fixes = stage_data.setdefault("fixes", [])
    note = f"Recovered execution-host connectivity after: {error}"
    if isinstance(fixes, list) and note not in fixes:
        fixes.append(note)
    atomic_write_json(report_path, report)


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
    * the experimenter stops being asked for numbers at all. It names the eval dir it stands
      behind (``stages.eval.summary_dir``) — a decision, since a node may have scored several
      checkpoints — and the numbers come from that dir's own summary.json. This replaces matching
      the dirs against a score the experimenter claimed, which existed only to catch a claim the
      contract no longer invites: a score that no eval produced now has nowhere to come from.
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

    # The experimenter names the dir it stands behind; we supply what that dir measured. When it
    # names one we do not have (a typo, a path from another node) the newest is the fallback, and
    # the mismatch is worth a line — but it is not worth refusing to stamp, because the numbers
    # from a real eval dir are still better than the nothing that would otherwise reach the leader.
    declared = eval_stage.get("summary_dir")
    winner = next((p for p in points if p["eval_dir"] == declared), None)
    if winner is None:
        winner = points[0]  # newest
        if isinstance(declared, str) and declared:
            log.warning(
                "experimenter[%s]: summary_dir %s is none of the %d scored eval dirs %s "
                "— stamping the newest instead",
                name, declared, len(points), [p["eval_dir"] for p in points],
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
        from .runtime_compose import task_stage_config

        train_owned_by_agent = task_stage_config(cfg, "train").get("ownership") == "agent"
        if isinstance(train, dict) and not train.get("model_path") and train_owned_by_agent:
            train["model_path"] = runtime._paths(cfg, study_id, name, exp_dir)["model"]
    _stamp_measurements(cfg=cfg, study_id=study_id, name=name, report=report)
    return report


def _failed_report(*, error: str) -> dict:
    """The report for a node that produced nothing — no stages, because none finished.

    It used to synthesise an eval stage of `{"status": "failed", "score": null, "limit": 0}`. Every
    part of that was a claim about a stage that never ran, and `limit: 0` in particular was a value
    no real eval can produce (-1 is full, positive is a sample) — which is how it became legible as
    a fingerprint: the four nodes lost on 2026-08-15 all carried it, copied verbatim out of the
    prompt block that mirrored this function. The run-level `error` says what happened; a stage
    that did not happen says nothing.
    """
    return {"id": _make_id("exp"), "status": "failed", "error": error}


def _numeric_score(data: dict) -> float | None:
    score = data.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def _score_of(partial: dict) -> float | None:
    """The partial report's score, or None if it cannot be read at all.

    Salvage is the LAST-RESORT path — it exists because something already went wrong — so the
    one thing it must not do is add a second failure. It did: reading the partial unguarded, it
    hit a leaf whose type an experimenter had filled with prose, and the ValidationError went up
    through ``fan_out``, through ``run_auto``, and out of the process. The run died with 17
    minutes of budget left, on a node that had a score, in the function whose entire purpose is
    to recover a score from a report that is not clean.

    ``read()`` is far more forgiving since the status cursor landed, but it can still raise on a
    document that is not report-shaped at all — and this is the one caller that must never let it.
    """
    try:
        return report_schema.read(partial).score
    except Exception as e:  # noqa: BLE001 — any unreadable partial means "no score here"
        log.warning("salvage: partial report.json is unreadable (%s); trying the eval dirs", e)
        return None


def _salvage_report(*, cfg, study_id: str, spec: dict, partial: dict | None,
                    error: str) -> dict | None:
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
                # Name the dir, not the numbers: _stamp_measurements reads it and fills the rest.
                report, source = (
                    {"stages": {"eval": {"summary_dir": eval_dir}}},
                    f"remote summary.json ({eval_dir})",
                )
                break
    if report is None:
        return None
    # The score IS real; the cut was operational. Moving the cursor to `completed` is the whole of
    # it — the previous version also had to FABRICATE a review stage (`failure_classes: {}`,
    # `anomalies: "unknown"`, `sampled: false`) because the old resume check inferred "the author
    # considered itself done" from a review being present, so a salvaged report without one read as
    # mid-run. That invention went to the leader verbatim, as a node that had reviewed its rollouts
    # and found no failure classes at all. A cursor says "done" by itself, so the lie is unnecessary
    # and gone; a `completed` report with no review now means exactly what it looks like.
    report["status"] = "completed"
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
    task_name = runtime_config.get("task", "posttrainbench")
    runtime_config[task_name] = {
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
    from . import runtime_compose, stages as _stages, train_spec as _train_spec

    bench = cfg.target_bench
    run_root = runtime._run_root(cfg, study_id, spec["name"])
    parent_model = (
        (spec.get("train") or {}).get("model_name_or_path")
        or spec.get("init_model")
        or cfg.h100_base_model
    )
    data = spec.get("data") or {}
    recipe = data
    runtime_block = _runtime_prompt_block(
        cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir, eval_limit=eval_limit,
    )
    env_ops, task_def, resolved = runtime_compose.render_sections(
        cfg, parent_model=parent_model, eval_limit=eval_limit,
    )
    stage_blocks = _stages.render_stage_blocks(_stages.build_stages(resolved))
    train = spec.get("train") or {}
    train_frozen = (
        resolved.get("task", {}).get("stages", {}).get("train", {}).get("ownership")
        == "frozen"
    )
    train_spec_json = ""
    if train.get("finetuning_type") and not train_frozen:  # our method; skip if frozen or baseline
        ts = _train_spec.build_train_spec(
            resolved, name=spec["name"], plan_train=train, parent_model=parent_model,
            run_root=run_root, time_h=2.0 if cfg.ptb_profile == "smoke" else 10.0,
        )
        train_spec_json = json.dumps(ts, ensure_ascii=False, indent=2)
    turn = runtime_compose.task_agent_config(cfg).get("turns") or {}
    return _prompt.render(
        turn.get("experimenter", "experimenter/turn/experiment"),
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
        iteration=spec.get("iteration"),
        max_iterations=spec.get("max_iterations"),
        remaining_iterations=(
            spec["max_iterations"] - spec["iteration"]
            if spec.get("max_iterations") is not None and spec.get("iteration") is not None
            else None
        ),
        suite_dir=spec.get("suite_dir"),
        suite_target=spec.get("suite_target"),
        suite_target_state=spec.get("suite_target_state"),
        suite_run_dir=spec.get("suite_run_dir"),
        submission_path=spec.get("submission_path"),
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
    pending_host = (
        _reopen_execution_host_wait(report_path, existing_report)
        if existing_report is not None
        else None
    )
    if (
        existing_report is not None
        and pending_host is None
        and report_schema.is_terminal(existing_report)
    ):
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
    # The model this NODE starts from, resolved the way the prompt resolves it — not
    # `cfg.h100_base_model`, which is the run's base and is the node's parent only for a root
    # node. A continuation node logged "base=<run base>" two lines under its own
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
        report = _report_from_disk(report_path)
        pending_host = (
            _reopen_execution_host_wait(report_path, report) if report is not None else None
        )
        if pending_host is not None:
            stage, error = pending_host
            state["status"] = "waiting_for_execution_host"
            state["last_error"] = error
            _write_state(state_path, state)
            if not _wait_for_execution_host(cfg, deadline=deadline):
                break
            report = _report_from_disk(report_path) or report
            _record_execution_host_recovery(report_path, report, stage, error)
            state["status"] = "running"
            state.pop("last_error", None)
            _write_state(state_path, state)

        remaining = None if deadline is None else max(
            0.0, deadline + runtime._queue_wait_s(cfg) - time.monotonic()
        )
        if remaining is not None and remaining <= 0:
            break
        try:
            res = agent.run(prompt, resume=state.get("session_id"), timeout=remaining)
        except Exception as e:  # noqa: BLE001 — transient transport drops are already retried
            # (reconnect+resume) inside dataos_agent via CliAgent.run; reaching here means a genuine
            # blocker or the task-selected execution deadline is spent, so record it and stop.
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
        pending_host = (
            _reopen_execution_host_wait(report_path, report) if report is not None else None
        )
        if pending_host is not None:
            prompt = _continue_prompt(
                state,
                last_text,
                remaining_minutes=int(remaining / 60) if remaining is not None else None,
            )
            continue
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
        err = "experimenter produced no terminal report.json before its execution stopped"
        if state.get("last_error"):
            err = f"{err}: {state['last_error']}"
        # the eval may have finished before the cut — rescue a real score if one exists
        salvaged = _salvage_report(cfg=cfg, study_id=study_id, spec=spec, partial=report, error=err)
        report = salvaged if salvaged is not None else _failed_report(error=err)
        # Only now: salvage reads the eval dirs, and an eval about to write its summary must not
        # be killed before it does. Past that, whatever this node still has running is running
        # for nobody — the round has moved on — and on a one-job QOS it holds the slot the next
        # node is waiting for. The status cursor makes this rarer; it does not make it never.
        if (killed := runtime.cancel_live_jobs(cfg, study_id, name)):
            log.info("experimenter[%s]: cancelled %d job(s) left running by the abandoned node: %s",
                     name, len(killed), ", ".join(killed))
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
    baseline_run_root = runtime._run_root(cfg, "round-0000", "baseline")
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
        # "Is the baseline report finished?" is the same question the experimenter loop asks, so it
        # gets the same answer from the same function. It used to be re-implemented here as a test
        # on a TOP-LEVEL `status` — a field the investigator's own prompt forbids writing, because
        # the report's status is derived. So the compliant investigator hung: healthbench's had
        # both deliverables on disk at 02:27 (106 source profiles, a baseline scoring 6.70) and was
        # still being told to finish them three hours later, while two runs whose investigators
        # ignored the instruction sailed through. A predicate no one is allowed to satisfy is worse
        # than no predicate.
        baseline_done = _read_json(baseline_report) or {}
        if isinstance(recon, dict) and recon.get("source_profiles") \
                and report_schema.is_terminal(baseline_done):
            log.info("recon: done — %d source profiles, base_cause=%s, baseline=%s",
                     len(recon.get("source_profiles") or []),
                     (recon.get("baseline_diagnosis") or {}).get("cause"),
                     report_schema.read(baseline_done).score)
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
    """Run each spec as an agentic experimenter. Experiments execute SEQUENTIALLY, sharing the
    run's task-selected budget. ``deadline`` is present only for wall-clock tasks and stops
    launching further experiments once that budget is spent."""
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
    log.info("fan_out: %d agentic experimenter(s), SEQUENTIAL bench=%s",
             len(specs), cfg.target_bench)
    reports: dict[str, dict] = {}
    for spec in specs:
        name = spec["name"]
        if deadline is not None and time.monotonic() >= deadline:
            log.info("fan_out: execution deadline spent — skipping experiment %s", name)
            # No stages: this node never started, so it has nothing to say about train or eval.
            reports[name] = {
                "id": _make_id("exp"), "name": name, "status": "failed",
                "error": "skipped: execution deadline exhausted",
            }
            continue
        _, report = _run_one(cfg, study_id, spec, exp_root, timeout=_exp_timeout(cfg, deadline))
        reports[name] = report
    return reports
