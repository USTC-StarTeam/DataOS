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


# Role -> the ORDER its context blocks compose in. Every block lives in
# src/dataos/prompt/<role>/context/ and belongs to exactly ONE role: a block shared by two
# roles has to be written to serve both, so neither gets the sharpest version. Membership is
# the directory's business (test_prompt_layers asserts the two agree); this table only fixes
# the order, which is a real decision — the hardest constraints go first, domain knowledge
# last, because a long context file buries whatever trails it.
_BLOCKS = {
    "leader": (
        "leader/context/role",
        "leader/context/search_discipline",
        "leader/context/operator_selection",
    ),
    "experimenter": (
        "experimenter/context/role",
        "experimenter/context/operator_use",
        "experimenter/context/operator_custom",
        "experimenter/context/training",
        "experimenter/context/cluster_discipline",
        "experimenter/context/eval_contract",
        "experimenter/context/report_contract",
        "experimenter/context/failure_diagnosis",
    ),
    "investigator": (
        "investigator/context/role",
        "investigator/context/pool_survey",
        "investigator/context/cluster_discipline",
        "investigator/context/eval_contract",
    ),
}


def prompt_facts(cfg: AutoRunConfig) -> dict:
    """The run-level facts a CONTEXT template may reference — and the only ones it may.

    A context block is written once per run into the CLI's context file, so it can only
    carry values that hold for the whole run. Anything that changes between turns
    (parent model, remaining minutes, the experiment's recipe) belongs in a turn template.
    ``test_prompt_layers`` enforces that by checking every context template's variables
    against these keys."""
    # Read defensively: this is now called for EVERY role, including from tests that pass a
    # sparse double with no budget/runtime_config.
    budget = getattr(cfg, "budget", None)
    budget_h = (getattr(budget, "max_wall_clock_s", None) or 0) / 3600 or 10
    budget_hours = int(budget_h) if budget_h == int(budget_h) else round(budget_h, 1)
    runtime = getattr(cfg, "runtime_config", None) or {}
    gres = str((runtime.get("cluster") or {}).get("gres", "gpu:1"))
    try:
        gpus = int(gres.rsplit(":", 1)[-1])
    except ValueError:
        gpus = 1
    # the eval policy as a NAME, so a context block can carry only the branch that applies
    # instead of explaining every mode and its magic number. Mirrors loop._effective_eval_limit.
    limit = getattr(cfg, "eval_limit", None) or (
        20 if getattr(cfg, "ptb_profile", "") == "smoke" else 200
    )
    cluster = runtime.get("cluster") or {}
    qos = cluster.get("qos")
    return {
        "budget_hours": budget_hours,
        "gpus": gpus,
        # Deployment facts a discipline block states as constraints. They arrive as VALUES so the
        # block carries the rule once and the numbers come from the machine — the same move
        # `eval_mode` below makes for the eval policy. Before this they were prose inside
        # env.yaml, which meant no golden test, no review, and one copy per environment.
        "qos": qos,
        # the choice's CONSEQUENCE, resolved here rather than re-derived in prose: only the
        # guarded QOS needs the memory-placeholder + idle-matmul prescription, and an environment
        # that carried both branches as text is what produced two near-identical env files.
        "has_gpu_guard": bool(qos) and qos == cluster.get("guard_qos"),
        "mem_per_gpu_mb": cluster.get("mem_per_gpu_mb") or 125000,
        "cpus_per_gpu": cluster.get("cpus_per_gpu") or 11,
        "gres_required": cluster.get("gres_required") or "gpu:1",
        "node_pool": cluster.get("node_pool") or "",
        "nodes_without_libcuda_link": cluster.get("nodes_without_libcuda_link") or "",
        # the single PTB bench this run optimizes; lets a context block name it instead of
        # carrying prose about benches this run will never touch
        "bench": getattr(cfg, "target_bench", None) or "the target benchmark",
        "eval_mode": "auto" if limit == -2 else ("full" if limit == -1 else "fixed"),
        "eval_limit": limit,
    }


def system_prompt(role: str, cfg: AutoRunConfig) -> str:
    """Compose the role's context prompt from src/dataos/prompt/<role>/context/."""
    if role not in _BLOCKS:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(_BLOCKS)}")
    from .runtime_compose import task_agent_config

    configured = task_agent_config(cfg).get("context") or {}
    blocks = tuple(configured.get(role) or _BLOCKS[role])
    # Every block gets the same facts: a template that ignores them renders identically, and
    # any block is then free to reference a fact without a special case here.
    facts = prompt_facts(cfg)
    return "\n\n".join(_prompt.render(b, **facts) for b in blocks)


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
            '"expected_signal":"completed stub report",'
            f'"subagent_brief":"run {name}"}}'
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

    The role's cwd comes from ``role_cwd`` (outside the repo, so the CLI
    cannot auto-discover the repo's development-time AGENTS.md). Only the run record
    stays here, under "_agent" — kept OUTSIDE the project dir because Project.studies
    lists every subdir of the project dir and would mis-count it as a study."""
    rec_dir = Path(cfg.workspace) / "_agent" / cfg.project_id / role
    rec_dir.mkdir(parents=True, exist_ok=True)

    if cfg.backend == "stub":
        # Only the leader role is ever built (loop.py / tests pass "leader").
        if role == "leader":
            return _StubMainAgent(cfg.stub_replies)
        raise ValueError(f"stub backend only supports role 'leader', got {role!r}")

    if cfg.backend == "cli":
        # A code-agent CLI with its own local login — no API key handled here.
        return build_agent(
            cfg,
            role=role,
            instructions=system_prompt(role, cfg),
            transcript=str(rec_dir / "transcript.jsonl"),  # the leader's run record
        )

    raise ValueError(f"unknown backend {cfg.backend!r}; use 'cli' or 'stub'")
