"""Monitor stall detection: a live run whose every agent transcript has gone
silent is wedged (CLI stall / dead stream) and must be flagged."""

import os
import time

from dataos.agent.monitor import _STALL_WINDOW, _stall_s


def test_stall_none_without_transcripts(tmp_path):
    run = tmp_path / "run-1"
    run.mkdir()
    assert _stall_s(tmp_path, "run-1") is None


def test_stall_tracks_the_freshest_role(tmp_path):
    run = tmp_path / "run-1"
    leader = tmp_path / "_agent" / "run-1" / "leader"
    leader.mkdir(parents=True)
    old = leader / "transcript.jsonl"
    old.write_text("{}\n")
    stale_ts = time.time() - 2 * _STALL_WINDOW
    os.utime(old, (stale_ts, stale_ts))
    assert _stall_s(tmp_path, "run-1") > _STALL_WINDOW  # only a stale leader tee

    exp = run / "round-0001" / "experiments" / "e1"
    exp.mkdir(parents=True)
    (exp / "agent.jsonl").write_text("{}\n")  # fresh experimenter tee wins
    assert _stall_s(tmp_path, "run-1") < 60
