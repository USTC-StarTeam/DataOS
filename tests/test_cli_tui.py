from __future__ import annotations

from typing import Any

from py_agent.cli import run
from py_agent.providers.base import ChatResult, Provider


class DummyProvider(Provider):
    def chat(self, messages: list[dict[str, Any]], model: str, tools=None) -> ChatResult:
        return ChatResult(model=model, text="ok", tool_calls=[])


def test_cli_tui_mode_routes_to_tui_runner(monkeypatch, test_workspace) -> None:
    monkeypatch.chdir(test_workspace)

    class DummyProviderFactory(DummyProvider):
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    captured: dict[str, Any] = {}

    def fake_tui_mode(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("py_agent.cli.OpenAICompatibleProvider", DummyProviderFactory)
    monkeypatch.setattr("py_agent.cli.run_tui_mode", fake_tui_mode)
    rc = run(["--mode", "tui"])
    assert rc == 0
    assert captured["model"] == "glm-5"
