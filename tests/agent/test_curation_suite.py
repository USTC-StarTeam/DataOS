import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "configs" / "tasks" / "curationbench" / "suite.py"


def _module(monkeypatch):
    spec = importlib.util.spec_from_file_location("curation_suite", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    package = types.ModuleType("benchmark")
    tools = types.ModuleType("benchmark.tools")
    suite = types.ModuleType("benchmark.tools.suite")
    suite.PENDING = "pending"
    suite.ACTIVE = "active"
    suite.COMPLETED = "completed"
    suite.FAILED = "failed"
    monkeypatch.setitem(sys.modules, "benchmark", package)
    monkeypatch.setitem(sys.modules, "benchmark.tools", tools)
    monkeypatch.setitem(sys.modules, "benchmark.tools.suite", suite)
    return module


class _Target:
    def __init__(self, iteration, state="pending"):
        self.task_id = "task"
        self.iteration = iteration
        self.state = state
        self.key = f"task_iter{iteration}"
        self.run_dir = Path(f"/suite/{self.key}")


class _Controller:
    def __init__(self, states):
        self.suite_dir = Path("/suite")
        self.targets = [_Target(i, state) for i, state in enumerate(states, 1)]
        self._active_index = next(
            (i for i, target in enumerate(self.targets) if target.state == "active"), None
        )
        self._resumed_active = self._active_index is not None

    @property
    def active_target(self):
        return self.targets[self._active_index] if self._active_index is not None else None

    def begin_next_target(self):
        active = self.active_target
        if active is not None and active.state == "active" and self._resumed_active:
            self._resumed_active = False
            return self._context(active)
        if active is not None and active.state == "active":
            active.state = "failed"
        target = next((target for target in self.targets if target.state == "pending"), None)
        if target is None:
            return {"status": "suite_completed"}
        target.state = "active"
        self._active_index = target.iteration - 1
        return self._context(target)

    @staticmethod
    def _context(target):
        return {
            "status": "target_activated",
            "target_key": target.key,
            "iteration": target.iteration,
        }


def _args(iteration=1, iterations=3):
    return types.SimpleNamespace(expected_iteration=iteration, iterations=iterations)


def test_claim_resumes_the_same_active_target_without_spending_another(monkeypatch):
    module = _module(monkeypatch)
    controller = _Controller(["active", "pending", "pending"])
    monkeypatch.setattr(module, "_ensure", lambda args: controller)

    result = module.claim(_args(1))

    assert result["iteration"] == 1
    assert [target.state for target in controller.targets] == ["active", "pending", "pending"]


def test_claiming_the_next_node_marks_an_unfinished_prior_target_failed(monkeypatch):
    module = _module(monkeypatch)
    controller = _Controller(["active", "pending", "pending"])
    monkeypatch.setattr(module, "_ensure", lambda args: controller)

    result = module.claim(_args(2))

    assert result["iteration"] == 2
    assert [target.state for target in controller.targets] == ["failed", "active", "pending"]


def test_completed_target_is_reconciled_without_advancing(monkeypatch):
    module = _module(monkeypatch)
    controller = _Controller(["completed", "pending", "pending"])
    monkeypatch.setattr(module, "_ensure", lambda args: controller)

    result = module.claim(_args(1))

    assert result["iteration"] == 1 and result["target_state"] == "completed"
    assert controller.targets[1].state == "pending"


@pytest.mark.parametrize("state,submitted,job_state,eligible", [
    ("active", None, "FAILED", True),
    ("active", None, "COMPLETED", True),
    ("completed", None, "FAILED", False),
    ("active", "/submission", "FAILED", False),
    ("active", None, "RUNNING", False),
    ("active", None, "TIMEOUT", False),
])
def test_preflight_check_is_read_only_and_requires_no_training(
    tmp_path, monkeypatch, state, submitted, job_state, eligible,
):
    module = _module(monkeypatch)
    project_root = tmp_path / "project" / "_curation_suite"
    suite = project_root / "suite"
    target = suite / "task_iter1"
    target.mkdir(parents=True)
    state_path = suite / "suite_state.json"
    state_path.write_text(json.dumps({"active_index": 0, "targets": [{
        "key": "task_iter1", "state": state, "iteration": 1, "task_id": "task",
        "run_dir": str(target),
    }]}))
    session_path = target / "session_state.json"
    session_path.write_text(json.dumps({
        "last_submission_path": submitted, "finetune_model_path": None,
        "_finetune_result": None, "_finetune_submission_path": None,
    }))
    data_root = project_root.parent / "round-0000" / "candidate"
    data_root.mkdir(parents=True)
    handle = data_root / "data_handle.json"
    handle.write_text(json.dumps({"launcher": "slurm", "job_id": "123", "run_dir": str(data_root)}))
    before = (state_path.read_bytes(), session_path.read_bytes())
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: f"123|{job_state}\n")
    args = types.SimpleNamespace(project_root=str(project_root), expected_iteration=1,
                                 task="task", data_handle=str(handle))
    assert module.preflight_check(args)["eligible"] is eligible
    assert before == (state_path.read_bytes(), session_path.read_bytes())
    if eligible:
        (target / "finetune" / "ft-in-progress").mkdir(parents=True)
        assert module.preflight_check(args)["reason"] == "training_attempt_present"


def test_finalize_refuses_pending_and_closes_a_terminal_queue(monkeypatch):
    module = _module(monkeypatch)
    pending = _Controller(["completed", "pending"])
    monkeypatch.setattr(module, "_ensure", lambda args: pending)
    with pytest.raises(RuntimeError, match="pending iterations"):
        module.finalize(_args(iterations=2))

    terminal = _Controller(["completed", "active"])
    monkeypatch.setattr(module, "_ensure", lambda args: terminal)
    result = module.finalize(_args(iterations=2))
    assert result["status"] == "suite_completed"
    assert terminal.targets[1].state == "failed"
