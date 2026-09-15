#!/usr/bin/env python3
"""Host-level supervision of one DataOS run; invoked by a user systemd service/timer.

The controller remains the normal DataOS CLI. Remote recovery only finishes already
submitted evaluations through dataos-eval; it never creates training experiments.
"""

from __future__ import annotations

import argparse
import fcntl
import json
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
            if service.get("ActiveState") == "failed":
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
