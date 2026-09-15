# Host-level DataOS watchdog

`scripts/watch_run.py` is a user-systemd companion to the existing DataOS CLI, not another
experiment executor. It supports three modes:

- `run`: derive the remaining budget from `STATE_DIR/intent.json`'s `deadline_epoch`,
  write `active_config.json`, then exec the normal DataOS CLI with `--resume`.
- `check`: inspect the service, heartbeat and this project's Slurm jobs; finish missing
  judge stages through the configured `dataos-eval collect`; record `latest.json` and
  append `history.jsonl`. Pass `--service` and `--timer` with the installed unit names.
- `remote`: the SSH-side implementation, restricted to the configured project's run root.

Pass the same absolute `--config` and `--state-dir` to the run/check services. Mirror the
private run config into the remote checkout's `configs/runs/` directory. The run must already
exist on the agent host because supervision always resumes it. The service can use a login
shell for existing authentication; never put API keys in unit files or this state directory.

Use a 30-minute timer plus an exact calendar trigger at the recorded deadline. The host must
remain powered on, with its user manager available. `latest.json` is the current status;
`needs_attention` or `stale_heartbeat` requires inspection. This host-level watchdog cannot
send notifications into a Codex chat and does not autonomously edit application code.

Recovery is bounded: the timer retries recognized transient controller failures at most three
times. A service may additionally use systemd's bounded `Restart=on-failure` policy. Restarts
never reset the deadline. Active GPU jobs are not cancelled merely because a controller restarts.
At deadline, or when the run contains `stopped.json`, the watcher stops the controller,
cancels only jobs whose working directories belong to this project, and stops its timer.

The evaluator owns a per-output `.judge.lock`, inherited by the judge process. This prevents
the controller and watchdog from scoring the same outputs concurrently. The watcher never
re-evaluates completed scores, changes the benchmark protocol, or submits training jobs.

To stop intentionally, stop/disable the watchdog timer and stop the controller service. Keep
the original artifacts and mark `stopped.json` if the timer might later be re-enabled.
