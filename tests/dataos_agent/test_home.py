"""Each CLI's context file name, and the env precedence chain.

Config-root isolation was removed after it broke ``codex``'s login: these CLIs keep an OAuth
credential with a single-use refresh token inside that root, so N private copies means the first
refresh invalidates every other one. See ``dataos_agent._home``.
"""

from __future__ import annotations

from dataos_agent._home import context_filename
from dataos_agent.providers.codex import CodexSpec
from dataos_agent.providers.kimi import KimiSpec
from dataos_agent.types import AgentConfig


def test_context_filename_per_provider():
    assert context_filename("codex") == "AGENTS.md"
    assert context_filename("kimi") == "AGENTS.md"
    assert context_filename("claude") == "CLAUDE.md"
    assert context_filename("qwen") == "QWEN.md"
    assert context_filename("fake") is None  # no context file -> caller keeps using the flag


def test_proxy_overrides_a_stale_inherited_value(monkeypatch):
    """A wrong *_PROXY exported in the shell would otherwise silently break the CLI."""
    monkeypatch.setenv("HTTPS_PROXY", "http://stale:1")
    env = CodexSpec().env(AgentConfig(provider="codex"))
    assert env["HTTPS_PROXY"] == CodexSpec.proxy


def test_caller_env_wins_over_the_provider_default():
    env = CodexSpec().env(
        AgentConfig(provider="codex", extra={"env": {"HTTPS_PROXY": "http://mine:2"}})
    )
    assert env["HTTPS_PROXY"] == "http://mine:2"


def test_env_is_none_when_nothing_to_override():
    """A provider with no proxy inherits os.environ untouched."""
    assert KimiSpec.proxy is None
    assert KimiSpec().env(AgentConfig(provider="kimi")) is None


def test_instructions_not_reprepended_on_resume():
    """The high-level prompt must not be re-sent every resumed turn: the session has it."""
    cfg = {"provider": "codex", "instructions": "ROLE BRIEF", "workspace": "/ws"}
    fresh = CodexSpec().build_argv(AgentConfig(**cfg), "task")
    resumed = CodexSpec().build_argv(AgentConfig(**cfg, extra={"resume": "tid-1"}), "task")
    assert fresh[-1] == "ROLE BRIEF\n\ntask"
    assert resumed[-1] == "task"


def test_codex_pins_project_doc_budget():
    """Codex truncates an over-budget context file silently, so the budget is pinned."""
    argv = CodexSpec().build_argv(AgentConfig(provider="codex"), "hi")
    assert any(a.startswith("project_doc_max_bytes=") for a in argv)


def test_kimi_instructions_not_reprepended_on_resume():
    cfg = {"provider": "kimi", "instructions": "BRIEF", "workspace": "/ws"}
    fresh = KimiSpec().build_argv(AgentConfig(**cfg), "task")
    resumed = KimiSpec().build_argv(AgentConfig(**cfg, extra={"resume": "s1"}), "task")
    assert "BRIEF\n\ntask" in fresh
    assert "task" in resumed and "BRIEF\n\ntask" not in resumed


def test_role_cwd_is_per_role_and_stable(tmp_path, monkeypatch):
    from dataos.agent.llm import config as _cfg
    from dataos.agent.llm.config import role_cwd

    monkeypatch.setattr(_cfg, "_AGENT_ROOT", tmp_path)  # don't touch the real ~/.cache
    a, b = role_cwd("proj-x", "leader"), role_cwd("proj-x", "experimenter")
    assert a == role_cwd("proj-x", "leader")  # stable -> resume works
    assert a != b                             # roles share no context file
