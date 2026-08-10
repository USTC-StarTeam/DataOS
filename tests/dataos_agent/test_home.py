"""Isolated config roots + the env precedence chain that carries them."""

from __future__ import annotations

from pathlib import Path

from dataos_agent._home import context_filename, home_env
from dataos_agent.providers.codex import CodexSpec
from dataos_agent.providers.kimi import KimiSpec
from dataos_agent.types import AgentConfig


def test_context_filename_per_provider():
    assert context_filename("codex") == "AGENTS.md"
    assert context_filename("kimi") == "AGENTS.md"
    assert context_filename("claude") == "CLAUDE.md"
    assert context_filename("qwen") == "QWEN.md"
    assert context_filename("fake") is None  # no context file -> caller keeps using the flag


def test_home_env_prepares_root_and_names_the_var(tmp_path):
    home = tmp_path / "codex-home"
    assert home_env("codex", str(home)) == {"CODEX_HOME": str(home)}
    assert home.is_dir()  # created eagerly: the CLI must find a usable root


def test_home_env_is_idempotent(tmp_path):
    home = str(tmp_path / "h")
    assert home_env("codex", home) == home_env("codex", home)


def test_home_env_unknown_provider_is_a_no_op(tmp_path):
    assert home_env("fake", str(tmp_path / "nope")) == {}


def test_credentials_are_copied_never_symlinked(tmp_path, monkeypatch):
    """A symlink would let the CLI's re-init of its credential store destroy the real
    login (kimi does exactly that), so the seed must be a copy the CLI can safely own."""
    import dataos_agent._home as _h

    real = tmp_path / "real-auth.json"
    real.write_text('{"token": "live"}', encoding="utf-8")
    monkeypatch.setitem(_h._SPEC, "codex", ("CODEX_HOME", "AGENTS.md", (str(real),)))

    home = tmp_path / "home"
    home_env("codex", str(home))
    seeded = home / "real-auth.json"
    assert seeded.is_file() and not seeded.is_symlink()
    assert seeded.read_text(encoding="utf-8") == '{"token": "live"}'

    # the CLI rewriting its copy must NOT reach the real file
    seeded.write_text('{"token": ""}', encoding="utf-8")
    home_env("codex", str(home))  # a later spawn must not clobber the refreshed copy
    assert real.read_text(encoding="utf-8") == '{"token": "live"}'
    assert seeded.read_text(encoding="utf-8") == '{"token": ""}'


def test_env_precedence_proxy_then_home_then_caller(tmp_path, monkeypatch):
    """os.environ < provider proxy < config root < extra["env"] (the caller wins)."""
    monkeypatch.setenv("HTTPS_PROXY", "http://stale:1")  # a wrong value in the shell
    home = str(tmp_path / "h")

    env = CodexSpec().env(AgentConfig(provider="codex", home=home))
    assert env["HTTPS_PROXY"] == CodexSpec.proxy  # OVERRIDES the stale inherited value
    assert env["CODEX_HOME"] == home

    env = CodexSpec().env(
        AgentConfig(provider="codex", home=home, extra={"env": {"HTTPS_PROXY": "http://mine:2"}})
    )
    assert env["HTTPS_PROXY"] == "http://mine:2"  # caller overrides the provider default


def test_env_is_none_when_nothing_to_override():
    """A provider with no proxy and no isolated root inherits os.environ untouched."""
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


def test_role_dirs_are_per_role_and_stable(tmp_path, monkeypatch):
    from dataos.agent.llm import config as _cfg
    from dataos.agent.llm.config import role_dirs

    monkeypatch.setattr(_cfg, "_AGENT_ROOT", tmp_path)  # don't touch the real ~/.cache
    cwd_a, home_a = role_dirs("proj-x", "leader")
    cwd_b, home_b = role_dirs("proj-x", "experimenter")
    assert (cwd_a, home_a) == role_dirs("proj-x", "leader")  # stable -> resume works
    assert cwd_a != cwd_b and home_a != home_b  # roles share no context file or session
    for p in (cwd_a, home_a, cwd_b, home_b):
        assert Path(p).is_dir()
