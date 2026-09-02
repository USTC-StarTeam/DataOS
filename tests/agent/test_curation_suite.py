import importlib.util
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
