"""Construct the agent for a role — the single place that knows about backend choice.

Roles: ``leader`` (the persistent plan/review dialog), ``experimenter`` (realizes one
experiment end to end), ``investigator`` (one-shot pre-training intel).

- ``backend == "cli"`` → a code-agent CLI (claude / kimi / qwen ..., chosen by
  ``cfg.agent_provider``) via dataos_agent, using the CLI's own local login — NO API
  key. See :mod:`dataos.agent.llm.cli_agent` and :func:`dataos.agent.llm.config.build_agent`.
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


# Role -> system-prompt assembly: skeleton + its kb knowledge blocks, in order.
# Each block is a YAML template in src/dataos/prompt/{system,kb}/; only the leader
# skeleton takes render params (the run's facts).
_BLOCKS = {
    "leader": ("system/leader", "kb/operator_selection", "kb/pool_survey", "system/strategy"),
    "experimenter": ("system/experimenter", "kb/operator_use", "kb/operator_custom",
                     "kb/training", "kb/cluster_discipline", "kb/eval_format",
                     "kb/report_contract", "kb/failure_diagnosis"),
    "investigator": ("system/investigator", "kb/pool_survey", "kb/cluster_discipline",
                     "kb/eval_format"),
}


def _run_facts(cfg: AutoRunConfig) -> dict:
    budget_h = (cfg.budget.max_wall_clock_s or 0) / 3600 or 10
    budget_hours = int(budget_h) if budget_h == int(budget_h) else round(budget_h, 1)
    gres = str((cfg.runtime_config.get("cluster") or {}).get("gres", "gpu:1"))
    try:
        gpus = int(gres.rsplit(":", 1)[-1])
    except ValueError:
        gpus = 1
    return {"budget_hours": budget_hours, "gpus": gpus}


def system_prompt(role: str, cfg: AutoRunConfig) -> str:
    """Compose the role's system prompt from src/dataos/prompt (skeleton + kb blocks)."""
    if role not in _BLOCKS:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(_BLOCKS)}")
    # only the leader skeleton takes render params; experimenter/investigator blocks are
    # all static, so sparse test doubles (no budget/runtime_config) work for them
    facts = _run_facts(cfg) if role == "leader" else {}
    return "\n\n".join(
        _prompt.render(b, **facts) if b == "system/leader" else _prompt.render(b)
        for b in _BLOCKS[role]
    )


class _StubMainAgent:
    """Tiny structured fake used by tests for the unified leader role."""

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
            return SimpleNamespace(text=text, output=None, session_id="stub-leader")
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
            session_id="stub-leader",
        )


def make_agent(role: str, cfg: AutoRunConfig):
    """Build the agent for ``role`` ("leader" | "experimenter" | "investigator").

    The leader produces no files directly, so its workspace lives outside the
    project dir; otherwise Project.studies would mis-count it as a study."""
    # kept OUTSIDE the project dir — Project.studies lists every subdir of the project
    # dir, so an "_agent" dir there would be mis-counted as a study.
    ws = str(Path(cfg.workspace) / "_agent" / cfg.project_id / role)
    Path(ws).mkdir(parents=True, exist_ok=True)

    if cfg.backend == "stub":
        # Only the leader role is ever built (loop.py / tests pass "leader").
        if role == "leader":
            return _StubMainAgent(cfg.stub_replies)
        raise ValueError(f"stub backend only supports role 'leader', got {role!r}")

    if cfg.backend == "cli":
        # A code-agent CLI with its own local login — no API key handled here.
        return build_agent(
            cfg,
            instructions=system_prompt(role, cfg),
            workspace=ws,
            transcript=str(Path(ws) / "transcript.jsonl"),  # the leader's run record
        )

    raise ValueError(f"unknown backend {cfg.backend!r}; use 'cli' or 'stub'")
