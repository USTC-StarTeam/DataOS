"""Exec-host facts for one run — the cluster and the on-disk layout, with no knowledge of agents.

Split out of the old ``subagent_executor``, which mixed two unrelated kinds of knowledge: how to
drive one agent to a terminal result, and where things live on the exec host. The tell that the
split was overdue is that ``loop.py`` reached THROUGH the executor for four of these helpers
(``_run_root``, ``_queue_wait_s``, ``_queue_credit_s``, ``set_invocation_wall_start``) — a caller
borrowing cluster facts from a module named after agent execution.

Deliberately NOT named after a task. PostTrainBench is one task among others
(``configs/tasks/<task>/``), and ssh probing, Slurm queue-credit accounting and run-root layout are
none of its business — a core module carrying one task's name in its filename would make every
future task look like a guest in it.

The PTB-shaped ``setdefault`` fallbacks in :func:`_runtime_config` are a deliberate exception, not
an oversight: this reader has to stay usable against the sparse config doubles the tests build, and
turning a missing key into a hard error there would trade a real convenience for a naming purity
nobody consumes. A second task supplies its own ``environment``/``task`` from the run config, which
is where the values properly come from; the defaults only decide what happens when nobody said.

Nothing here builds, prompts, or resumes an agent. Both :mod:`dataos.agent.roles` and
:mod:`dataos.agent.loop` import it.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from dataos.core.study import _slug

#: What a runtime config must carry, BY KIND. The set is not universal: ssh coordinates and a
#: Slurm partition are facts about reaching another machine, and a workstation run has neither —
#: demanding them there would be demanding proof of a cluster that is deliberately absent. The
#: environment declares its own kind (env.yaml `kind:`), so this is the same rule as everywhere
#: else in this repo: configuration decides, code only enforces what was declared.
_REQUIRED_CLUSTER_KEYS: dict[str, tuple[str, ...]] = {
    "slurm": ("host", "user", "port", "key", "partition", "qos", "gres"),
    "local": ("gres",),  # still says how many devices this run may take
}
_REQUIRED_PATH_KEYS = (
    "deploy_root", "runs_base", "fixed_pool", "train_env", "eval_env",
)  # conda_sh dropped — envs are uv venvs (called via {path}/bin/<tool>)
_REQUIRED_GATEWAY_KEYS = ("base_url", "key_file")  # connection facts only; judge model is the task's



def _runtime_config(cfg) -> dict:
    """Return the runtime config (paths / cluster / gateway) with legacy fallbacks for
    sparse test doubles. Production cfgs carry everything from configs/environments."""
    runtime = json.loads(json.dumps(getattr(cfg, "runtime_config", {}) or {}))
    paths = runtime.setdefault("paths", {})
    runtime.setdefault("cluster", {})
    runtime.setdefault("environment", "example-slurm")
    runtime.setdefault("task", "posttrainbench")
    paths.setdefault("runtime_root", f"{paths.get('deploy_root', '')}/configs")
    return runtime

def _task_assets_dir(runtime: dict) -> str:
    """The runtime task dir on the exec host (eval bridge & co). Prompt sections come from
    the runtime composer (runtime_compose) — no playbook files are read anymore."""
    root = runtime.get("paths", {}).get("runtime_root", "")
    task = runtime.get("task", "posttrainbench")
    return f"{root}/tasks/{task}"

def _missing_runtime_config_keys(cfg) -> list[str]:
    runtime = _runtime_config(cfg)
    kind = str(runtime.get("kind") or "slurm")
    required = _REQUIRED_CLUSTER_KEYS.get(kind)
    if required is None:
        return [f"kind={kind!r} is not a known environment kind "
                f"(known: {sorted(_REQUIRED_CLUSTER_KEYS)})"]
    missing = [
        f"cluster.{key}" for key in required
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
    runs_base = _runtime_config(cfg).get("paths", {}).get("runs_base", "")
    return f"{runs_base}/{cfg.project_id}/{study_id}/{_slug(str(name))}"

def _ssh_text(cfg, remote_cmd: str, *, timeout: int = 25) -> str:
    """Run a read-only remote command over ssh; '' on any failure (best-effort probes —
    a measurement failure must never break the run)."""
    import os
    import subprocess

    cluster = _runtime_config(cfg).get("cluster") or {}
    if not cluster.get("host"):
        return ""
    try:
        return subprocess.run(
            [
                "ssh", "-i", os.path.expanduser(str(cluster.get("key", ""))),
                "-p", str(cluster.get("port", 22)),
                "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=8",
                f"{cluster.get('user')}@{cluster['host']}", remote_cmd,
            ],
            capture_output=True, text=True, timeout=timeout,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _exec_text(cfg, cmd: str, *, timeout: int = 25) -> str:
    """Run a command where the JOBS are: over ssh on a cluster, in a shell on a workstation.

    ``_ssh_text`` above returns "" the moment a deployment declares no host, which is correct
    for a PROBE (a measurement that fails is not a run that fails) but wrong for anything that
    has to actually happen. A workstation's exec host is this machine.
    """
    import subprocess

    cluster = _runtime_config(cfg).get("cluster") or {}
    if cluster.get("host"):
        return _ssh_text(cfg, cmd, timeout=timeout)
    try:
        return subprocess.run(
            ["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def exec_host_command(cfg, argv: list[str], *, timeout: int = 120) -> str:
    """Run one checked command on the configured execution host.

    Unlike the best-effort probes above, task lifecycle transitions must either return their
    output or stop loudly. ``argv`` remains structured until the final SSH boundary.
    """
    import os
    import subprocess

    cluster = _runtime_config(cfg).get("cluster") or {}
    command = list(argv)
    if cluster.get("host"):
        command = [
            "ssh",
            "-i",
            os.path.expanduser(str(cluster.get("key", ""))),
            "-p",
            str(cluster.get("port", 22)),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=8",
            f"{cluster.get('user')}@{cluster['host']}",
            shlex.join(argv),
        ]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        detail = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(
            f"execution-host command failed ({proc.returncode}): {detail[-4000:]}"
        )
    return proc.stdout.strip()


#: Stop every job this run_dir still has in flight. Runs ON the exec host, where the handles and
#: the scheduler are, in one round trip.
#:
#: Cancellation is a LAUNCHER operation — ``Stage.cancel`` is a staticmethod because it needs
#: nothing from the stage — so one loop handles the data build, the probe, the training run and
#: every checkpoint's eval alike, whichever instrument submitted them.
#:
#: It can enumerate them at all only because each job writes ``<job_name>.handle.json``: until
#: 2026-08-14 a run_dir held ONE ``handle.json``, overwritten per submit, so a node's earlier
#: jobs were unreachable from any path.
_CANCEL_LIVE = """
import glob, json
from dataos_launch.registry import get_launcher
from dataos_launch.types import Handle
cancelled, seen = [], set()
for path in sorted(glob.glob({run_dir!r} + "/*handle.json")):
    try:
        handle = Handle.model_validate(json.load(open(path)))
    except Exception:
        continue          # dataos-eval's own record is not Handle-shaped; its launcher one is
    if handle.job_id in seen:
        continue          # handle.json duplicates whichever job submitted last
    seen.add(handle.job_id)
    launcher = get_launcher(handle.launcher)
    if launcher.status(handle).is_terminal():
        continue          # already over; cancelling would be a lie in the log
    launcher.cancel(handle)
    cancelled.append(handle.job_id)
print(json.dumps(cancelled))
"""


def cancel_live_jobs(cfg, study_id: str, name) -> list[str]:
    """Kill what a node left running, and say which ids. Best-effort: never raises.

    Called when the runtime gives up on a node. Nothing did this before, so an abandoned node's
    jobs ran on: two trainings shared one card on the workstation, and on the cluster a dead
    node's data build held ``debug_normal``'s only job slot for seven minutes while the next
    node waited for it.
    """
    py = f"{(_runtime_config(cfg).get('paths') or {}).get('sft_env', '')}/bin/python"
    script = _CANCEL_LIVE.format(run_dir=_run_root(cfg, study_id, name))
    out = _exec_text(cfg, f"{shlex.quote(py)} -c {shlex.quote(script)}")
    try:
        ids = json.loads(out or "[]")
    except json.JSONDecodeError:
        return []
    return [str(i) for i in ids] if isinstance(ids, list) else []


_INVOCATION_WALL_START: float | None = None  # wall-clock start of THIS run_auto invocation

def set_invocation_wall_start(ts: float) -> None:
    """Anchor queue-credit accounting: only queue waits overlapping THIS invocation count
    (budget is per-invocation wall-clock; without the anchor every --resume re-credits the
    whole history)."""
    global _INVOCATION_WALL_START
    _INVOCATION_WALL_START = ts

def _queue_credit_s(rows: list[tuple[float, float]], *, since: float) -> float:
    """Sum the [submit, wait_end] overlap with [since, +inf) over (submit, wait_end) epochs."""
    return sum(max(0.0, end - max(sub, since)) for sub, end in rows)

def _queue_wait_s(cfg) -> float:
    """Total Slurm queue-wait time across this run's jobs, in seconds, overlapping THIS
    invocation.

    The run's budget tracks IDEAL run time — queue waiting is budget-free by design
    (cluster contention is environment noise, not experiment work). Job ids are found via
    slurm-<id>.out logs and *handle.json files (dataos-train handle.json + dataos-eval
    eval_handle.json) under runs_base/<project_id>; sacct epochs are resolved cluster-side.
    A job that never started is capped at its End (cancellation), never at now — a
    cancelled-pending job must not print credit forever."""
    import shlex

    runtime = _runtime_config(cfg)
    runs_base = (runtime.get("paths") or {}).get("runs_base")
    if not (runtime.get("cluster") or {}).get("host") or not runs_base:
        return 0.0
    base = shlex.quote(f"{runs_base}/{cfg.project_id}")  # project ids can contain parens
    remote = (
        f"ids=$( (find {base} -maxdepth 4 -name 'slurm-*.out' 2>/dev/null | grep -oP 'slurm-\\K[0-9]+';"
        f" find {base} -maxdepth 3 -name '*handle.json' 2>/dev/null | xargs -r grep -hoP '\"job_id\":\\s*\"?\\K[0-9]+') | sort -u | tr '\\n' ',' );"
        " ids=${ids%,}; [ -z \"$ids\" ] && exit 0;"
        " now=$(date +%s);"
        # -X: main job rows only (no .batch/.extern steps)
        " sacct -j \"$ids\" --format=Submit,Start,End -P -n -X 2>/dev/null | while IFS='|' read -r sub sta end; do"
        "   se=$(date -d \"$sub\" +%s 2>/dev/null || echo 0); [ \"$se\" = 0 ] && continue;"
        "   ee=0;"
        "   for cand in \"$sta\" \"$end\"; do"  # Start first; a never-started job falls to End
        "     if [ -n \"$cand\" ] && [ \"$cand\" != None ] && [ \"$cand\" != Unknown ]; then"
        "       ee=$(date -d \"$cand\" +%s 2>/dev/null || echo 0);"
        "     fi;"
        "     [ \"$ee\" != 0 ] && break;"
        "   done;"
        "   [ \"$ee\" = 0 ] && ee=$now;"  # still pending: wait is still accruing
        "   [ $ee -lt $se ] && ee=$se;"
        "   echo \"$se $ee\";"
        " done"
    )
    out = _ssh_text(cfg, remote)
    rows: list[tuple[float, float]] = []
    for line in (out or "").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return _queue_credit_s(rows, since=_INVOCATION_WALL_START or 0.0)

def _paths(cfg, study_id: str, name, exp_dir: Path) -> dict:
    """The DETERMINISTIC artifact layout for one experiment (H100 RUN_ROOT + local report).

    Recorded in ``state.json`` for monitoring and resume, NOT in report.json: seven paths the
    leader cannot open were 19.3% of every report's bytes.

    The eval dir is deliberately absent. It used to be built here as
    ``eval/{bench}_limit{"auto" if eval_limit == -2 else eval_limit}``, and ``auto`` is a tag no
    eval has ever written — so the one consumer that resolved a real directory from it (score
    salvage) looked in a path that does not exist. Naming a directory you did not create is a
    guess; :func:`_remote_eval_dirs` asks instead.
    """
    rr = _run_root(cfg, study_id, name)
    return {
        "run_root": rr,
        "data": f"{rr}/data/sft",
        "adapter": f"{rr}/adapter",
        "model": f"{rr}/final_model",
        "logs": f"{rr}/logs",
        "report": str(exp_dir / "report.json"),
    }

def _remote_eval_summaries(cfg, study_id: str, name) -> list[tuple[str, dict]]:
    """Every ``summary.json`` under this run's ``eval/``, newest first.

    Asks the exec host what is there rather than reconstructing the directory name. A checkpoint
    sweep leaves several eval dirs, an AUTO-sized eval leaves one whose tag nobody can predict from
    ``eval_limit``, and salvage only has to find A real score — newest is the best guess at the one
    the experimenter was writing when it was cut off.
    """
    rr = _run_root(cfg, study_id, name)
    out = _ssh_text(cfg, f"ls -1dt {shlex.quote(rr)}/eval/*/summary.json 2>/dev/null")
    found: list[tuple[str, dict]] = []
    for path in (out or "").split():
        text = _ssh_text(cfg, f"cat {shlex.quote(path)} 2>/dev/null")
        try:
            summary = json.loads(text) if text else None
        except json.JSONDecodeError:
            summary = None
        if isinstance(summary, dict):
            found.append((str(Path(path).parent), summary))
    return found
