#!/usr/bin/env python3
"""Host-level supervision of one DataOS run; invoked by a user systemd service/timer.

The controller remains the normal DataOS CLI. Remote recovery only finishes already
submitted evaluations through dataos-eval; it never creates training experiments.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def command(argv: list[str], timeout: int = 30) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"{argv[0]} exited {result.returncode}: {result.stderr[-800:]}")
    return result.stdout.strip()


def owned_job(workdir: str, root: Path) -> bool:
    return bool(workdir) and Path(workdir).resolve().is_relative_to(root.resolve())


def remote_check(config: dict, env: dict, cancel: bool) -> dict:
    """Only inspect/cancel this project's jobs; finish pending judge stages once."""
    root = Path(config["exec"]["runs_base"]) / Path(config["workspace"]).name
    if root == Path(config["exec"]["runs_base"]) or not root.is_absolute():
        raise ValueError("invalid project run root")
    jobs = []
    for job_id in command(["squeue", "-u", env["access"]["user"], "-h", "-o", "%i"]).split():
        if not re.fullmatch(r"\d+(?:_\d+)?", job_id):
            continue
        try:
            info = command(["scontrol", "show", "job", job_id, "-o"])
        except RuntimeError:
            continue  # the job may have completed since squeue
        match = re.search(r"\bWorkDir=(\S+)", info)
        if match and owned_job(match[1], root):
            state = re.search(r"\bJobState=(\S+)", info)
            jobs.append({"job_id": job_id, "state": state[1] if state else "UNKNOWN"})
    if cancel:
        if jobs:
            command(["scancel", *[j["job_id"] for j in jobs]])
        return {"cancelled": jobs}

    from dataos_eval.stage import EvalStage
    from dataos_launch import Handle, Status

    collectors = []
    providers = [t for t in env["tools"].values() if "dataos-eval" in t.get("provides", [])]
    if len(providers) != 1:
        raise ValueError("expected exactly one dataos-eval provider")
    cli = str(Path(providers[0]["path"]) / "bin/dataos-eval")
    for path in sorted(root.glob("round-*/*/eval_*.eval_handle.json")):
        record = read_json(path)
        if not record.get("needs_judge") or not record.get("launch"):
            continue
        output = Path(record["output_dir"])
        if not owned_job(str(output), root):
            raise ValueError("eval output escapes this project's run root")
        summary = read_json(output / "summary.json")
        if summary.get("score") is not None or not (output / "predictions.jsonl").is_file():
            continue
        if EvalStage.status(Handle.model_validate(record["launch"])) != Status.COMPLETED:
            continue
        with (output / ".judge.lock").open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                collectors.append({"handle": str(path), "state": "already_judging"})
                continue
        argv = [
            cli, "collect", "--repo", config["exec"]["deploy_root"],
            "--environment", config["environment"], "--task-config", config["task"], str(path),
        ]
        with (output / "watchdog_judge.log").open("a") as log:
            process = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "POSTTRAINBENCH_HEALTHBENCH_JUDGE_LOG":
                     str(output / "judge_events.jsonl")},
            )
        collectors.append({"handle": str(path), "state": "collector_started", "pid": process.pid})
    return {"jobs": jobs, "judges": collectors}


def probe_remote(config: dict, env: dict, source: Path, cancel: bool = False) -> dict:
    access = env["access"]
    deploy = Path(config["exec"]["deploy_root"])
    python = str(Path(env["tools"]["sft_env"]["path"]) / "bin/python")
    remote = [python, str(deploy / "scripts/watch_run.py"), "remote", "--config",
              str(deploy / "configs/runs" / source.name)]
    if cancel:
        remote.append("--cancel")
    stdout = command([
        "ssh", "-i", str(Path(access["key_ref"]).expanduser()), "-p", str(access["port"]),
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
        f"{access['user']}@{access['host']}", shlex.join(remote),
    ], timeout=120)
    return json.loads(stdout.splitlines()[-1])


def run_controller(config: dict, state_dir: Path) -> None:
    remaining = float(read_json(state_dir / "intent.json")["deadline_epoch"]) - time.time()
    if remaining <= 0:
        return
    config["budget"]["time_hours"] = remaining / 3600
    active = state_dir / "active_config.json"
    write_json(active, config)
    os.chdir(REPO)
    os.execv(str(REPO / ".venv/bin/dataos"), ["dataos", str(active), "--resume"])


def completed_evidence(config: dict, env: dict, live: dict, raw: dict) -> dict:
    """Read authoritative evaluation evidence before fixing a completion report."""
    stages = raw.get("stages", {})
    summary_dir = stages.get("eval", {}).get("summary_dir", "")
    model = stages.get("train", {}).get("model_path", "")
    project = Path(config["exec"]["runs_base"]) / Path(config["workspace"]).name
    root = Path(live.get("node_run_root") or "/")
    if not owned_job(str(root), project) or not owned_job(summary_dir, root):
        return {}
    if live.get("node_trains") and not owned_job(model, root):
        return {}
    access = env["access"]
    program = (
        "import json,pathlib,sys; "
        "p=pathlib.Path(sys.argv[1])/'summary.json'; "
        "print(json.dumps({'summary': json.loads(p.read_text()), "
        "'model_exists': pathlib.Path(sys.argv[2]).is_dir() if sys.argv[2] else False}))"
    )
    remote = shlex.join(["python3", "-c", program, summary_dir, model])
    output = command([
        "ssh", "-i", str(Path(access["key_ref"]).expanduser()), "-p", str(access["port"]),
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=15",
        f"{access['user']}@{access['host']}", remote,
    ])
    return json.loads(output)


def repair_report(config: dict, env: dict, live: dict, path: Path) -> bool:
    from dataos.agent.report import is_terminal, read

    original = path.read_text()
    raw = json.loads(original)
    if raw.get("status") not in {"complete", "completed"}:
        return False
    changed = raw["status"] == "complete"
    raw["status"] = "completed"
    for stage in raw.get("stages", {}).values():
        if isinstance(stage, dict) and isinstance(stage.get("fixes"), str):
            stage["fixes"] = [stage["fixes"]]
            changed = True
    if not changed:
        return False
    parsed = read(raw)  # Do not guess how to repair other schema errors.
    if not is_terminal(raw) or parsed.score is None:
        return False
    evidence = completed_evidence(config, env, live, raw)
    summary = evidence.get("summary", {})
    score = summary.get("score")
    if (
        not isinstance(score, (int, float)) or not math.isfinite(score)
        or not math.isclose(parsed.score, score, abs_tol=0.0051)
        or summary.get("needs_judge") is not False
        or not summary.get("n_samples")
        or summary["n_samples"] != raw["stages"]["eval"].get("n_samples")
        or (live.get("node_trains") and not evidence.get("model_exists"))
    ):
        return False
    if path.read_text() != original:
        return False  # The experimenter advanced while the SSH probe was in flight.
    digest = hashlib.sha256(original.encode()).hexdigest()[:12]
    path.with_name(f"report.before-watchdog-{digest}.json").write_text(original)
    write_json(path, raw)
    return True


def check_progress(config: dict, env: dict, args, workspace: Path, live: dict, state: dict):
    """Heartbeats and repeated prose are not evidence of experiment progress."""
    from dataos.agent.report import read

    now = time.time()
    round_idx = live.get("round")
    paths = (
        sorted((workspace / f"round-{round_idx:04d}" / "experiments").glob("*/report.json"))
        if isinstance(round_idx, int) else []
    )
    path = paths[0] if len(paths) == 1 and owned_job(str(paths[0]), workspace) else None
    active_work = bool(state["remote"].get("jobs") or state["remote"].get("judges"))
    report = None
    error = None
    if path:
        try:
            if not active_work and repair_report(config, env, live, path):
                state["actions"].append("repaired_report_schema")
            report = read_json(path)
            read(report)
            if report.get("status") == "complete":
                error = "Unrecognized terminal status: complete; completion evidence unverified"
        except Exception as exc:
            error = str(exc)[:1500]
    if error:
        state["report_error"] = error
        state["status"] = "invalid_report"

    # Use tool requests, not repeated acknowledgments or polling results/timestamps.
    tools_seen = []
    if path and (path.parent / "agent.jsonl").exists():
        with (path.parent / "agent.jsonl").open("rb") as stream:
            stream.seek(max(0, stream.seek(0, 2) - 128_000))
            lines = stream.read().decode(errors="replace").splitlines()
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "tool_use":
                tools_seen.append(json.dumps([event.get("name"), event.get("args")], sort_keys=True))
    tools_seen = sorted(set(tools_seen))
    marker = hashlib.sha256(json.dumps([
        live.get("node_id"), live.get("phase"), report, tools_seen,
    ], sort_keys=True).encode()).hexdigest()
    progress_path = args.state_dir / "progress.json"
    prior = read_json(progress_path)
    advanced = prior.get("marker") != marker
    last_progress = now if advanced or active_work else prior.get("last_progress_at", now)
    write_json(progress_path, {"marker": marker, "last_progress_at": last_progress})
    state["progress"] = {
        "node_id": live.get("node_id"), "last_progress_at": last_progress,
        "idle_seconds": max(0, now - last_progress), "remote_work_active": active_work,
    }
    if active_work or state["service"].get("ActiveState") != "active":
        return
    if now - last_progress < 40 * 60:
        return

    counts_path = args.state_dir / "stall_restarts.json"
    counts = read_json(counts_path)
    key = str(live.get("node_id") or live.get("phase"))
    if counts.get(key, 0) == 0 and not error:
        counts[key] = 1
        write_json(counts_path, counts)
        command(["systemctl", "--user", "restart", args.service], timeout=60)
        write_json(progress_path, {"marker": marker, "last_progress_at": now})
        state["actions"].append("restarted_stalled_controller")
        state["status"] = "recovering"
    else:
        command(["systemctl", "--user", "stop", args.service], timeout=60)
        state["status"] = "needs_attention"
        state["actions"].append("paused_unresolved_stall")
        write_json(args.state_dir / "attention.json", {
            "detected_at": now, "node_id": live.get("node_id"),
            "reason": error or "No progress after one controller recovery; no remote work active",
        })


def check(config: dict, env: dict, args) -> None:
    from dataos_agent._retry import is_recoverable_text

    state_dir = args.state_dir
    intent = read_json(state_dir / "intent.json")
    workspace = Path(config["workspace"])
    if not workspace.is_absolute():
        workspace = REPO / workspace
    live = read_json(workspace / "live.json")
    state = {"checked_at": time.time(), "phase": live.get("phase"), "actions": []}
    service = dict(line.split("=", 1) for line in command([
        "systemctl", "--user", "show", args.service, "-p", "ActiveState", "-p", "Result",
    ]).splitlines() if "=" in line)
    state["service"] = service
    expired = time.time() >= float(intent["deadline_epoch"])
    stopped = (workspace / "stopped.json").is_file()
    try:
        if expired or stopped:
            command(["systemctl", "--user", "stop", args.service])
            state["remote"] = probe_remote(config, env, args.config, cancel=True)
            command(["systemctl", "--user", "stop", args.timer])
            state["status"] = "budget_finished" if expired else "user_stopped"
        else:
            state["remote"] = probe_remote(config, env, args.config)
            state["heartbeat_age_s"] = time.time() - float(live.get("heartbeat", 0))
            state["status"] = "running"
            attention = read_json(state_dir / "attention.json")
            if attention:
                state["status"] = "needs_attention"
                state["attention"] = attention
            elif service.get("ActiveState") == "failed":
                log = workspace.parent / "_logs" / workspace.name / "run.log"
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 6000))
                    tail = stream.read().decode(errors="replace")
                prior = read_json(state_dir / "restart_count.json").get("count", 0)
                if prior < 3 and is_recoverable_text(tail):
                    write_json(state_dir / "restart_count.json", {"count": prior + 1})
                    command(["systemctl", "--user", "reset-failed", args.service])
                    command(["systemctl", "--user", "start", args.service])
                    state["actions"].append("resumed_controller")
                else:
                    state["status"] = "needs_attention"
            elif service.get("ActiveState") == "inactive":
                state["status"] = "controller_finished"
            elif state["heartbeat_age_s"] > 300:
                state["status"] = "stale_heartbeat"
            if not attention and service.get("ActiveState") == "active":
                check_progress(config, env, args, workspace, live, state)
    except Exception as exc:
        state["status"] = "needs_attention"
        state["error"] = str(exc)
    write_json(state_dir / "latest.json", state)
    with (state_dir / "history.jsonl").open("a") as log:
        log.write(json.dumps(state) + "\n")
    print(json.dumps(state))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "check", "remote"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--service")
    parser.add_argument("--timer")
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args()
    config = read_json(args.config)
    env = yaml.safe_load((REPO / "configs/environments" / config["environment"] / "env.yaml").read_text())
    if args.mode == "remote":
        print(json.dumps(remote_check(config, env, args.cancel)))
    elif args.mode == "run":
        run_controller(config, args.state_dir)
    else:
        check(config, env, args)


if __name__ == "__main__":
    main()
