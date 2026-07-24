"""Agentic per-experiment executor — a sub-agent IMPLEMENTS one tree node.

The main-agent fixes WHAT each experiment is (parent model, data intent, and SFT
hyperparameters). This module hands each experiment to an agentic Claude Code sub-agent that
figures out HOW to realize it — run the data recipe, submit training to H100 (ssh/sbatch,
configured Slurm), run the frozen official PostTrainBench eval, and write ``report.json`` —
self-healing around env/network/sbatch/operator failures.

Same ``fan_out(cfg, study_id, specs) -> {name: report}`` contract the loop used for the old
deterministic Slurm backend, so the loop swaps executors with one line. PTB allows only ONE
H100 for the whole run, so experiments run SEQUENTIALLY (never in parallel) under a shared
single-GPU / 10h budget; ``report.json`` on disk is the source of truth.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from dataos.core.ids import _make_id
from dataos.core.io import atomic_write_json
from dataos.core.study import _slug

from dataos import prompt as _prompt
from dataos.agent.llm.config import build_agent

log = logging.getLogger(__name__)

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
_RUNS_BASE = "/data/hcgu/users/tjh/dataos_agentver_runs"  # legacy fallback; prefer runtime_config
# Let the executor run FREE within its time budget. The real bound is the per-experiment
# DEADLINE (the timeout on each agent.run); if the sub-agent stops before writing report.json
# the resume loop continues the SAME session. A high cap means max_turns ~never chops a live
# experiment into artificial 150-turn pieces (which only added latency + a failure surface).
SUBAGENT_MAX_STEPS = 100_000
_EXP_TIMEOUT_SMOKE_S = 2 * 3600.0  # hard per-experiment timeout: 2 h (smoke)
_EXP_TIMEOUT_FORMAL_S = 10 * 3600.0  # PTB hard cap: 10 h / 1 H100
_STATE_FILE = "subagent_state.json"
_REQUIRED_CLUSTER_KEYS = ("host", "user", "port", "key", "partition", "qos", "gres")
_REQUIRED_PATH_KEYS = (
    "deploy_root", "runs_base", "fixed_pool", "train_env", "eval_env",
)  # conda_sh dropped — envs are uv venvs (called via {path}/bin/<tool>)
_REQUIRED_GATEWAY_KEYS = ("base_url", "model", "key_file")


def _exp_timeout(cfg, deadline: float | None) -> float:
    """Hard per-experiment timeout: a profile cap, never exceeding the remaining wall-clock."""
    cap = _EXP_TIMEOUT_SMOKE_S if cfg.ptb_profile == "smoke" else _EXP_TIMEOUT_FORMAL_S
    if deadline is not None:
        return max(60.0, min(cap, deadline - time.monotonic()))
    return cap


def _runtime_config(cfg) -> dict:
    """Return the runtime config (paths / cluster / gateway), defaulting runs_base."""
    runtime = json.loads(json.dumps(getattr(cfg, "runtime_config", {}) or {}))
    paths = runtime.setdefault("paths", {})
    runtime.setdefault("cluster", {})
    paths.setdefault("runs_base", _RUNS_BASE)
    # DataOS-Runtime split: default the new keys so pre-split runtime_configs still work.
    runtime.setdefault("environment", "PTB_H100_slurm")
    runtime.setdefault("task", "posttrainbench")
    paths.setdefault("runtime_root", paths.get("deploy_root", ""))
    return runtime


def _task_assets_dir(runtime: dict) -> str:
    """The runtime task dir on the exec host (eval bridge & co). Prompt sections come from
    the runtime composer (runtime_compose) — no playbook files are read anymore."""
    root = runtime.get("paths", {}).get("runtime_root", "")
    task = runtime.get("task", "posttrainbench")
    return f"{root}/tasks/{task}"


def _missing_runtime_config_keys(cfg) -> list[str]:
    runtime = _runtime_config(cfg)
    missing = [
        f"cluster.{key}" for key in _REQUIRED_CLUSTER_KEYS
        if not runtime.get("cluster", {}).get(key)
    ]
    missing.extend(
        f"paths.{key}" for key in _REQUIRED_PATH_KEYS
        if not runtime.get("paths", {}).get(key)
    )
    missing.extend(
        f"gateway.{key}" for key in _REQUIRED_GATEWAY_KEYS
        if not runtime.get("gateway", {}).get(key)
    )
    return missing


def _run_root(cfg, study_id: str, name) -> str:
    runs_base = _runtime_config(cfg).get("paths", {}).get("runs_base", _RUNS_BASE)
    return f"{runs_base}/{cfg.project_id}/{study_id}/{_slug(str(name))}"


def _artifacts(cfg, study_id: str, name, exp_dir: Path, eval_limit: int) -> dict:
    """The DETERMINISTIC artifact layout for one experiment (H100 RUN_ROOT + local report).
    Fixed by the runtime, recorded on the tree node — not left to the sub-agent's whim."""
    rr = _run_root(cfg, study_id, name)
    return {
        "run_root": rr,
        "data": f"{rr}/data/sft",
        "adapter": f"{rr}/adapter",
        "model": f"{rr}/final_model",
        "eval": f"{rr}/eval/{cfg.target_bench}_limit{eval_limit}",
        "logs": f"{rr}/logs",
        "report": str(exp_dir / "report.json"),
    }


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


def _stamp_report(*, cfg, study_id: str, spec: dict, exp_dir: Path, eval_limit: int,
                  report: dict) -> dict:
    """Attach runtime-owned provenance to any report the sub-agent wrote."""
    name = spec["name"]
    report.setdefault("id", _make_id("exp"))
    report.setdefault("name", name)
    report["artifacts"] = _artifacts(cfg, study_id, name, exp_dir, eval_limit)
    # The sub-agent writes its own training (LLaMA-Factory) and may record the commands it used
    # under report["scripts"]; the runtime only guarantees the keys exist for provenance.
    scripts = report.setdefault("scripts", {})
    scripts.setdefault("train", "")
    scripts.setdefault("data", "")
    # the sub-agent writes report.json freeform; "model" may be a bare path string,
    # a dict, or missing — normalize to {"path": ...} before stamping.
    model = report.get("model")
    if isinstance(model, str):
        model = {"path": model}
    elif not isinstance(model, dict):
        model = {}
    report["model"] = model
    if not model.get("path"):
        model["path"] = report["artifacts"]["model"]
    return report


def _failed_report(*, cfg, study_id: str, spec: dict, exp_dir: Path, eval_limit: int,
                   error: str) -> dict:
    report = {
        "id": _make_id("exp"),
        "name": spec["name"],
        "status": "failed",
        "score": None,
        "error": error,
        "stages": {"train": {"status": "failed"}, "eval": {"status": "failed"}},
    }
    return _stamp_report(
        cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
        eval_limit=eval_limit, report=report,
    )


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
        "artifacts": _artifacts(cfg, study_id, name, exp_dir, eval_limit),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _runtime_prompt_block(*, cfg, study_id: str, spec: dict, exp_dir: Path,
                          eval_limit: int) -> dict:
    runtime = _runtime_config(cfg)
    runtime.setdefault("paths", {})["run_root"] = _run_root(cfg, study_id, spec["name"])
    runtime["paths"]["local_report"] = str(exp_dir / "report.json")
    runtime["paths"]["local_state"] = str(exp_dir / _STATE_FILE)
    parent_model = (
        (spec.get("train") or {}).get("model_name_or_path")
        or spec.get("init_model")
        or cfg.h100_base_model
    )
    runtime["model"] = {
        "base_model": parent_model,
        "parent_model": parent_model,
        "root_base_model": cfg.h100_base_model,
        "base_model_id": getattr(cfg, "base_model_id", None),
    }
    runtime["posttrainbench"] = {
        "bench": cfg.target_bench,
        "profile": cfg.ptb_profile,
        "eval_limit": eval_limit,
    }
    if getattr(cfg, "runtime_config_path", None):
        runtime["source"] = str(cfg.runtime_config_path)
    return runtime


def _task_prompt(*, cfg, study_id: str, spec: dict, exp_dir: Path, eval_limit: int,
                 remaining_minutes: int | None = None) -> str:
    bench = cfg.target_bench
    run_root = _run_root(cfg, study_id, spec["name"])
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
        "subagent_task",
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
        "subagent_continue",
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
    eval_limit = getattr(cfg, "eval_limit", None)
    if eval_limit is None:
        eval_limit = 20 if cfg.ptb_profile == "smoke" else 200
    existing_report = _report_from_disk(report_path)
    if existing_report is not None and existing_report.get("status") in {"completed", "failed"}:
        report = _stamp_report(
            cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
            eval_limit=eval_limit, report=existing_report,
        )
        atomic_write_json(report_path, report)
        log.info("subagent[%s]: existing terminal report status=%s score=%s",
                 name, report.get("status"), report.get("score"))
        return name, report

    agent = build_agent(
        cfg,
        instructions=_prompt.render("subagent"),
        workspace=_REPO_ROOT,  # cwd at repo root so it can run .venv/ssh tooling
        max_steps=max(cfg.max_steps, SUBAGENT_MAX_STEPS),
    )
    log.info("subagent[%s]: starting (bench=%s base=%s timeout=%ss)",
             name, cfg.target_bench, cfg.h100_base_model, int(timeout) if timeout else None)
    state = _read_json(state_path) or _state_seed(
        cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
        report_path=report_path, eval_limit=eval_limit,
    )
    state.setdefault("artifacts", _artifacts(cfg, study_id, name, exp_dir, eval_limit))
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
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
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
            log.error("subagent[%s]: agent error: %s", name, e)
            break

        last_text = (getattr(res, "text", "") or "").strip()
        sid = getattr(res, "session_id", None)
        if sid:
            state["session_id"] = sid
        state["status"] = "running"
        state["last_message"] = last_text[-4000:]
        _write_state(state_path, state)
        log.info("subagent[%s]: message (%d chars, session=%s): %s", name, len(last_text),
                 state.get("session_id"), last_text[:800] + ("…" if len(last_text) > 800 else ""))

        report = _report_from_disk(report_path)
        if report is not None and report.get("status") in {"completed", "failed"}:
            state["status"] = report.get("status")
            _write_state(state_path, state)
            break

        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if remaining is not None and remaining <= 0:
            break
        log.info("subagent[%s]: no report.json yet; resuming same session", name)
        _sleep_before_resume(cfg, remaining if remaining is not None else 30.0)
        prompt = _continue_prompt(
            state, last_text,
            remaining_minutes=int(remaining / 60) if remaining is not None else None,
        )

    report = _report_from_disk(report_path)
    if report is None or report.get("status") not in {"completed", "failed"}:
        err = "sub-agent produced no terminal report.json before the PTB/experiment deadline"
        if state.get("last_error"):
            err = f"{err}: {state['last_error']}"
        report = _failed_report(
            cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
            eval_limit=eval_limit, error=err,
        )
    else:
        report = _stamp_report(
            cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
            eval_limit=eval_limit, report=report,
        )
    atomic_write_json(report_path, report)
    log.info("subagent[%s]: status=%s score=%s artifacts.run_root=%s",
             name, report.get("status"), report.get("score"), report["artifacts"]["run_root"])
    return name, report


_RECON_FILE = "recon.json"
_RECON_TIMEOUT_S = 45 * 60.0  # recon is cheap intel-gathering, not training — cap it tight


def run_recon_subagent(*, cfg, proj, catalog=None, deadline: float | None = None) -> dict:
    """RECON stage — a one-shot sub-agent that gathers ground-truth intel BEFORE any training and
    writes ``recon.json`` (eval_format_contract + baseline_diagnosis + source_profiles). It does NOT
    train; it ssh's to H100, reads the eval scorer, and runs the deterministic source profiler over
    the whole pool. Returns the parsed recon dict (also written to ``proj.project_dir/recon.json``).
    Resume-safe: if a valid recon.json already exists on disk, the loop skips calling this."""
    del catalog  # recon needs the pool + eval code, not the operator catalog
    recon_path = Path(proj.project_dir) / _RECON_FILE
    runtime = _runtime_config(cfg)
    paths = runtime.get("paths", {})
    cluster = runtime.get("cluster", {})
    deploy_root = paths.get("deploy_root", "")
    eval_env = paths.get("eval_env", "")
    task_dir = _task_assets_dir(runtime)
    prompt = _prompt.render(
        "subagent_recon",
        target_bench=cfg.target_bench,
        base_model=cfg.h100_base_model,
        fixed_pool=paths.get("fixed_pool", ""),
        deploy_root=deploy_root,
        eval_env_python=f"{eval_env}/bin/python" if eval_env else "python",
        cluster_json=json.dumps(
            {k: cluster.get(k) for k in ("host", "user", "port", "key")}, ensure_ascii=False
        ),
        profile_script=f"{deploy_root}/resources/recon/{runtime.get('task', 'posttrainbench')}/profile_source.py",
        lm_bridge=f"{task_dir}/eval",
        recon_path=str(recon_path),
    )
    agent = build_agent(
        cfg,
        instructions=_prompt.render("subagent"),
        workspace=_REPO_ROOT,
        max_steps=max(cfg.max_steps, SUBAGENT_MAX_STEPS),
    )
    cap = _RECON_TIMEOUT_S
    if deadline is not None:
        cap = max(60.0, min(cap, deadline - time.monotonic()))
    log.info("recon: starting reconnaissance (bench=%s, cap=%ss)", cfg.target_bench, int(cap))
    started = time.monotonic()
    session_id = None
    last_text = ""
    while time.monotonic() - started < cap:
        remaining = cap - (time.monotonic() - started)
        try:
            res = agent.run(prompt, resume=session_id, timeout=remaining)
        except Exception as e:  # noqa: BLE001 — recon must not crash the run; fall back to minimal
            log.error("recon: agent error: %s", e)
            break
        session_id = getattr(res, "session_id", None) or session_id
        last_text = (getattr(res, "text", "") or "").strip()
        recon = _read_json(recon_path)
        if isinstance(recon, dict) and recon.get("source_profiles"):
            log.info("recon: done — %d source profiles, base_cause=%s",
                     len(recon.get("source_profiles") or []),
                     (recon.get("baseline_diagnosis") or {}).get("cause"))
            return recon
        prompt = (
            "recon.json is not yet complete at %s. Finish gathering the three parts "
            "(eval_format_contract, baseline_diagnosis, source_profiles) and write the file."
            % recon_path
        )
    # fallback: don't block the run — write a minimal recon so the loop can proceed (loosely).
    log.warning("recon: incomplete (%s); writing minimal recon so the run can proceed", last_text[:200])
    recon = _read_json(recon_path) if isinstance(_read_json(recon_path), dict) else {}
    recon.setdefault("eval_format_contract", {"bench": cfg.target_bench, "summary": "recon incomplete"})
    recon.setdefault("baseline_diagnosis", {"cause": "unknown", "evidence": "recon incomplete"})
    recon.setdefault("source_profiles", [])
    atomic_write_json(recon_path, recon)
    return recon


def fan_out(*, cfg, study_id: str, specs: list, deadline: float | None = None) -> dict[str, dict]:
    """Run each spec as an agentic sub-agent. PTB allows only ONE H100 for the whole run, so
    experiments execute SEQUENTIALLY (never in parallel), sharing the single-GPU / 10h budget.
    ``deadline`` (monotonic) stops launching further experiments once the budget is spent."""
    if not cfg.target_bench:
        raise ValueError("subagent executor needs cfg.target_bench (the single PTB bench)")
    if not cfg.h100_base_model:
        raise ValueError("subagent executor needs cfg.h100_base_model (the -Base model path)")
    missing_runtime = _missing_runtime_config_keys(cfg)
    if missing_runtime:
        raise ValueError(
            "subagent executor needs runtime config with: " + ", ".join(missing_runtime)
        )
    exp_root = Path(cfg.workspace) / cfg.project_id / study_id / "experiments"
    log.info("fan_out: %d agentic sub-agent(s), SEQUENTIAL (PTB: 1 H100 at a time) bench=%s",
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
