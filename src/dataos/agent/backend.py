"""Construct the agent for a role — the single place that knows about backend choice.

- ``backend == "claude_code"`` → local Claude Code via the CLI's own Claude-account
  (subscription) login — NO API key. Used for the persistent main-agent dialog and
  execution sub-agents. See :mod:`dataos.agent.llm.claude_code` and
  :func:`dataos.agent.llm.config.build_agent`.
- ``backend == "stub"``  → a deterministic in-process fake (:class:`_StubMainAgent`)
  that returns fixed structured plans/reviews for tests; no network, no external package.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dataos import prompt as _prompt

from .config import AutoRunConfig
from .llm._output import parse_output
from .llm.config import build_agent


def _system_prompt(role: str, cfg: AutoRunConfig) -> str:
    """Render the role's system prompt from src/dataos/prompt, injecting the run's facts."""
    if role == "subagent":
        return _prompt.render("subagent")
    budget_h = (cfg.budget.max_wall_clock_s or 0) / 3600 or 10
    budget_hours = int(budget_h) if budget_h == int(budget_h) else round(budget_h, 1)
    gres = str((cfg.runtime_config.get("cluster") or {}).get("gres", "gpu:1"))
    try:
        gpus = int(gres.rsplit(":", 1)[-1])
    except ValueError:
        gpus = 1
    # the main-agent system prompt = base role + the maintainable strategy playbook
    return (
        _prompt.render("main_agent", budget_hours=budget_hours, gpus=gpus)
        + "\n\n"
        + _prompt.render("strategy")
    )


class _StubMainAgent:
    """Tiny structured fake used by tests for the unified main-agent role."""

    supports_dialog = False

    def __init__(self, replies: dict[str, Any]) -> None:
        self.replies = dict(replies)
        self.plan_count = 0
        self.review_count = 0

    def _reply(self, *keys: str) -> str | None:
        for key in keys:
            if key not in self.replies:
                continue
            value = self.replies[key]
            if isinstance(value, list):
                if len(value) > 1:
                    return str(value.pop(0))
                return str(value[0]) if value else "{}"
            return str(value)
        return None

    def _default_plan(self) -> str:
        name = f"stub-child-{self.plan_count}"
        self.plan_count += 1
        return (
            '{"parent_node_id":"n0000",'
            f'"name":"{name}",'
            f'"hypothesis":"stub experiment {name}",'
            '"estimated_hours":0.01,'
            '"data":{},'
            '"train":{"mode":"stub"},'
            '"eval_plan":"stub eval",'
            '"expected_signal":"completed stub report",'
            f'"subagent_brief":"run {name}",'
            '"rationale":"default stub plan"}'
        )

    def _default_review(self) -> str:
        self.review_count += 1
        return (
            '{"interpretation":"stub node completed",'
            '"lesson":"stub loop is healthy",'
            '"recommendation_for_future_growth":"continue with another small experiment"}'
        )

    def run(self, task: str, *, output=None, resume=None) -> SimpleNamespace:  # noqa: ARG002
        if output is None:
            text = self.replies.get("raw", "{}")
            return SimpleNamespace(text=text, output=None, session_id="stub-main-agent")
        name = getattr(output, "__name__", "")
        if name == "ReviewOut":
            text = self._reply(
                "review",
                "reviewer",
                "main_agent_review",
            ) or self._default_review()
        else:
            text = self._reply(
                "plan",
                "planner",
                "main_agent_plan",
            ) or self._default_plan()
        return SimpleNamespace(
            text=text,
            output=parse_output(text, output),
            session_id="stub-main-agent",
        )


def make_agent(role: str, cfg: AutoRunConfig):
    """Build the agent for ``role`` ("main_agent" | legacy "planner"/"reviewer").

    The main-agent produces no files directly, so its workspace lives outside the
    project dir; otherwise Project.studies would mis-count it as a study."""
    # kept OUTSIDE the project dir — Project.studies lists every subdir of the project
    # dir, so an "_agent" dir there would be mis-counted as a study.
    ws = str(Path(cfg.workspace) / "_agent" / cfg.project_id / role)
    Path(ws).mkdir(parents=True, exist_ok=True)

    if cfg.backend == "stub":
        # Only the unified main-agent role is ever built (loop.py / tests pass "main_agent").
        if role == "main_agent":
            return _StubMainAgent(cfg.stub_replies)
        raise ValueError(f"stub backend only supports role 'main_agent', got {role!r}")

    if cfg.backend == "claude_code":
        # Local Claude Code with the CLI's own Claude-account (subscription) login — no API key.
        return build_agent(cfg, instructions=_system_prompt(role, cfg), workspace=ws)

    raise ValueError(f"unknown backend {cfg.backend!r}; use 'claude_code' or 'stub'")
