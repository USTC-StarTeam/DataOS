from __future__ import annotations

import json

from dataos.core.sandbox import Sandbox


def test_sandbox_prepare_creates_expected_directories(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    for name in ["inputs", "outputs", "artifacts", "logs", "tmp", "code"]:
        assert (sandbox.root / name).is_dir()


def test_sandbox_write_json_and_log(tmp_path):
    sandbox = Sandbox(tmp_path / "sandbox")
    sandbox.prepare()

    report_path = sandbox.write_json("nested/result.json", {"ok": True})
    log_path = sandbox.log("run", "hello")

    assert json.loads(report_path.read_text(encoding="utf-8")) == {"ok": True}
    assert log_path.read_text(encoding="utf-8") == "hello"
    assert sandbox.report_path() == sandbox.root / "report.json"
