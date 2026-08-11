"""The stage probe's remote command must survive a real shell parse.

A ``run_root`` routinely contains ``(N)``: dataos appends it when a workspace name is taken, so
most real runs live under ``ptb-<bench>-4b2(1)`` / ``(2)``. An unquoted ``RR=...(2)...`` is a bash
SYNTAX error — the probe dies before printing ``PROBE_END``, and ``_parse_stage`` reports that as
``reachable: False``, indistinguishable from an unreachable cluster. The node name in the path is
leader-chosen, so the same interpolation is an injection surface.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess

import pytest

from dataos.agent import monitor

_PAREN = "/data2/hcgu/dataos_agentver_runs/ptb-gsm8k-4b2(2)/round-0001/gsm-verified-lora"


def _probe_for(monkeypatch, run_root: str) -> str:
    """The remote command stage_snapshot would send, without touching ssh."""
    sent: list[str] = []
    monkeypatch.setattr(monitor, "_ssh_read", lambda cmd, **kw: sent.append(cmd) or None)
    monitor._STAGE_CACHE.clear()
    monitor.stage_snapshot(run_root)
    assert sent, "stage_snapshot did not build a probe"
    return sent[0]


def _bash_parses(cmd: str) -> bool:
    """bash -n parses without executing — a real parse, not a shlex approximation."""
    return subprocess.run(["bash", "-n", "-c", cmd], capture_output=True).returncode == 0


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash to parse-check")
@pytest.mark.parametrize(
    "run_root",
    [
        _PAREN,
        "/runs/plain/round-0000/node",
        "/runs/ptb-x-4b2(1)/round-0002/node with space",
    ],
)
def test_probe_parses_under_bash(monkeypatch, run_root):
    assert _bash_parses(_probe_for(monkeypatch, run_root))


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash to parse-check")
def test_the_unquoted_form_really_does_fail(monkeypatch):
    """Negative control: without quoting, the same path is a bash syntax error — so the test
    above is proving something."""
    assert not _bash_parses(f"RR={_PAREN}; echo PROBE_END")


def test_path_round_trips_through_the_quoting(monkeypatch):
    cmd = _probe_for(monkeypatch, _PAREN)
    assert f"RR={shlex.quote(_PAREN)}" in cmd
    assert '"$RR"' in cmd  # expansions quoted too, so a space in the path cannot split it
    assert f"RR={_PAREN};" not in cmd  # never the bare form


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash to execute the probe")
def test_a_shell_metacharacter_in_the_node_name_cannot_escape(monkeypatch, tmp_path):
    """Node names are leader-chosen and unconstrained; quoting is the only thing between a
    hostile-looking name and arbitrary remote execution.

    Proven by RUNNING the probe (it only ls/echoes, so it is safe locally) and checking the
    payload never fired. Asserting on tokens would not prove it — the payload legitimately
    appears inside the quoted RR value.
    """
    marker = tmp_path / "pwned"
    hostile = f'/runs/x/round-0000/node";touch {marker};#'
    cmd = _probe_for(monkeypatch, hostile)
    done = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    assert "PROBE_END" in done.stdout  # the probe still completed
    assert not marker.exists(), "the node name escaped its quotes and executed"
    assert shlex.quote(hostile) in cmd
