"""Agentic per-experiment executor — a sub-agent IMPLEMENTS one tree edge.

The main-agent (loop + planner) fixes WHAT each experiment is (data operator recipe + SFT
hyperparameters) and which nodes to expand. This module hands each experiment to an agentic
Claude Code sub-agent (same brain as the loop, via the gateway) that figures out HOW to
realize it — run the operator recipe to build the data, submit training to H100 (ssh/sbatch,
configured Slurm), run the frozen official PostTrainBench eval, and write ``report.json`` — self-healing
around env/network/sbatch failures. It does NOT change the recipe or the eval protocol.

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

from dataos.agent.prompts import SUBAGENT_SYSTEM
from dataos.integrations.llm.claude_code import ClaudeCodeAgent

log = logging.getLogger(__name__)

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
_RUNS_BASE = "/data/hcgu/users/tjh/dataos_agentver_runs"  # legacy fallback; prefer runtime_config
SUBAGENT_MAX_STEPS = 150  # an executor needs many turns (data → sbatch → poll → eval → report)
_EXP_TIMEOUT_SMOKE_S = 1800.0  # hard per-experiment timeout: 30 min (smoke)
_EXP_TIMEOUT_FORMAL_S = 10 * 3600.0  # PTB hard cap: 10 h / 1 H100
_STATE_FILE = "subagent_state.json"
_REQUIRED_CLUSTER_KEYS = ("host", "user", "port", "key", "partition", "qos", "gres")
_REQUIRED_PATH_KEYS = (
    "deploy_root", "runs_base", "fixed_pool", "hf_cache", "conda_sh", "train_env", "eval_env",
)
_REQUIRED_GATEWAY_KEYS = ("base_url", "model", "key_file")


def _exp_timeout(cfg, deadline: float | None) -> float:
    """Hard per-experiment timeout: a profile cap, never exceeding the remaining wall-clock."""
    cap = _EXP_TIMEOUT_SMOKE_S if cfg.ptb_profile == "smoke" else _EXP_TIMEOUT_FORMAL_S
    if deadline is not None:
        return max(60.0, min(cap, deadline - time.monotonic()))
    return cap


def _runtime_config(cfg) -> dict:
    """Return runtime config with legacy --slurm values folded in for compatibility."""
    runtime = json.loads(json.dumps(getattr(cfg, "runtime_config", {}) or {}))
    paths = runtime.setdefault("paths", {})
    cluster = runtime.setdefault("cluster", {})
    legacy = getattr(cfg, "slurm", {}) or {}

    legacy_path_keys = {
        "agentver": "deploy_root",
        "deploy_root": "deploy_root",
        "runs_base": "runs_base",
        "fixed_pool": "fixed_pool",
        "hf_cache": "hf_cache",
        "conda_sh": "conda_sh",
        "train_env": "train_env",
        "eval_env": "eval_env",
    }
    for old_key, new_key in legacy_path_keys.items():
        if old_key in legacy and new_key not in paths:
            paths[new_key] = legacy[old_key]

    for key in ("user", "host", "port", "key", "partition", "qos", "gres"):
        if key in legacy and key not in cluster:
            cluster[key] = legacy[key]

    paths.setdefault("runs_base", _RUNS_BASE)
    return runtime


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


def _train_cmd(cfg, spec: dict, artifacts: dict) -> str:
    """The exact, deterministic train_sft.py command for this experiment (recipe → command),
    recorded on the tree node so training is fully reproducible. The method (lora|full) and all
    hyperparameters come from the main-agent's recipe; train_sft.py just executes them."""
    t = spec.get("train") or {}
    ft = t.get("finetuning_type", "lora")
    return (
        "python $AGENTVER/resources/code/posttrainbench/harness/train_sft.py "
        f"--base-model {cfg.h100_base_model} --data-dir {artifacts['data']} "
        f"--output-dir {artifacts['adapter']} --merged-dir {artifacts['model']} "
        f"--finetuning-type {ft} "
        f"--max-len {t.get('cutoff_len', 1024)} --bs {t.get('per_device_train_batch_size', 4)} "
        f"--grad-accum {t.get('gradient_accumulation_steps', 8)} "
        f"--lr {t.get('learning_rate', 1.5e-4)} --epochs {t.get('num_train_epochs', 1)} "
        f"--lora-r {t.get('lora_rank', 64)} --lora-alpha {t.get('lora_alpha', 128)}"
    )


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
    scripts = report.setdefault("scripts", {})
    scripts["train"] = _train_cmd(cfg, spec, report["artifacts"])
    scripts.setdefault("data", "")
    model = report.setdefault("model", {})
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
    runtime["model"] = {
        "base_model": cfg.h100_base_model,
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


def _task_prompt(*, cfg, study_id: str, spec: dict, exp_dir: Path, eval_limit: int) -> str:
    bench = cfg.target_bench
    run_root = _run_root(cfg, study_id, spec["name"])
    recipe = {"sources": (spec.get("data") or {}).get("sources", []),
              "recipe": (spec.get("data") or {}).get("recipe", [])}
    return "\n".join([
        f"Implement experiment {spec['name']!r} (bench={bench}, profile={cfg.ptb_profile}).",
        f"Base model (use this exact -Base, never instruct): {cfg.h100_base_model}",
        f"eval limit: {eval_limit}    smoke→cap data to ≤1000 rows" if cfg.ptb_profile == "smoke"
        else f"eval limit: {eval_limit}",
        "",
        "DATA RECIPE (fixed by the main-agent — run it exactly, do not change it):",
        json.dumps(recipe, ensure_ascii=False),
        "SFT hyperparameters (fixed):",
        json.dumps(spec.get("train") or {}, ensure_ascii=False),
        "",
        "RUNTIME CONFIG (authoritative — use these values instead of any examples in PLAYBOOK):",
        json.dumps(
            _runtime_prompt_block(
                cfg=cfg, study_id=study_id, spec=spec, exp_dir=exp_dir,
                eval_limit=eval_limit,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        "",
        f"H100 scratch for this experiment: RUN_ROOT={run_root}",
        f"Write the report to this LOCAL path (absolute): {exp_dir / 'report.json'}",
        f"Keep durable execution state at this LOCAL path: {exp_dir / _STATE_FILE}",
        "",
        "Read resources/code/posttrainbench/PLAYBOOK.md for the exact commands, paths, "
        "conda envs, ssh, eval scripts, and the report.json contract, then execute it: build "
        "data → train on H100 → eval → write report.json. Self-heal execution failures; ALWAYS "
        "write a report at the end (a failed report is still required). If a Slurm job is still "
        "running, keep polling it to terminal state; do not return a final prose-only update.",
    ])


def _continue_prompt(state: dict, last_text: str) -> str:
    return "\n".join([
        "Continue the SAME experiment. Do not re-plan and do not change the recipe.",
        f"report.json is still missing at: {state['report_path']}",
        f"execution state path: {state['state_path']}",
        "",
        "Current runtime state:",
        json.dumps(state, ensure_ascii=False),
        "",
        "Your previous message was:",
        last_text[-2000:] if last_text else "(no previous assistant text captured)",
        "",
        "Keep going until you write report.json with status completed or failed. If you already "
        "submitted Slurm jobs, poll sacct/squeue with sleeps to terminal state, then continue to "
        "the next stage. A stage update is not a final answer.",
    ])


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
    eval_limit = cfg.data_limit if cfg.data_limit is not None else (
        20 if cfg.ptb_profile == "smoke" else 200
    )
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

    agent = ClaudeCodeAgent(
        instructions=SUBAGENT_SYSTEM,
        workspace=_REPO_ROOT,  # cwd at repo root so it can read the playbook + run .venv/ssh
        model=cfg.claude_code_model,
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
                          eval_limit=eval_limit)
    last_text = ""
    while True:
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        if remaining is not None and remaining <= 0:
            break
        try:
            res = agent.run(prompt, resume=state.get("session_id"), timeout=remaining)
        except Exception as e:  # noqa: BLE001 — record the blocker; PTB time/user stop ends it
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
        prompt = _continue_prompt(state, last_text)

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
            "subagent executor needs --runtime-config with: " + ", ".join(missing_runtime)
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
