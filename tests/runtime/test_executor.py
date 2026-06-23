"""Executor fan-out over real run_one.py subprocesses with stub train/eval (no GPU/LLM).

Exercises the authoritative-report.json polling and skip_completed resume path.
"""

import json
from pathlib import Path

import dataos
from dataos.runtime import executor

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _setup(tmp_path, names):
    ws = str(tmp_path / "workspace")
    proj = dataos.project(id="p", workspace=ws)
    study = proj.study("round-0000")
    specs = [
        {
            "name": n,
            "data": {},
            "train": {"mode": "stub"},
            "eval": {"provider": "stub", "provider_config": {"score": 0.5}},
        }
        for n in names
    ]
    study.set_plan({"goal": "t", "experiments": specs})
    data = tmp_path / "data.jsonl"
    data.write_text('{"instruction":"a","input":"","output":"b"}\n', encoding="utf-8")
    return ws, specs, str(data)


def test_fanout_runs_stub_parallel(tmp_path):
    ws, specs, data = _setup(tmp_path, ["e1", "e2", "e3"])
    reports = executor.fan_out(
        project_id="p",
        study_id="round-0000",
        workspace=ws,
        specs=specs,
        data_dir=data,
        gpus=["0", "1"],  # concurrency 2 (value unused by stub)
        cwd=REPO_ROOT,
        poll_interval_s=0.05,
        skip_completed=True,
    )
    assert set(reports) == {"e1", "e2", "e3"}
    assert all(r["status"] == "completed" for r in reports.values())
    assert all(r["eval"]["score"] == 0.5 for r in reports.values())


def test_fanout_skip_completed(tmp_path):
    ws, specs, data = _setup(tmp_path, ["e1"])
    first = executor.fan_out(
        project_id="p", study_id="round-0000", workspace=ws, specs=specs,
        data_dir=data, gpus=[], cwd=REPO_ROOT, poll_interval_s=0.05,
    )
    rp = Path(ws) / "p" / "round-0000" / "experiments" / "e1" / "report.json"
    before = json.loads(rp.read_text())
    second = executor.fan_out(
        project_id="p", study_id="round-0000", workspace=ws, specs=specs,
        data_dir=data, gpus=[], cwd=REPO_ROOT, poll_interval_s=0.05, skip_completed=True,
    )
    after = json.loads(rp.read_text())
    assert first["e1"]["status"] == second["e1"]["status"] == "completed"
    assert after["id"] == before["id"]  # not re-run
