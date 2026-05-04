from __future__ import annotations

import httpx
import pytest

from dataos.llm.openai_compat import (
    DASHSCOPE_BASE_URL,
    DEFAULT_MODEL,
    OpenAICompatLLMClient,
)
from dataos.runtime.runner import _build_client_from_env


_REAL_HTTPX_CLIENT = httpx.Client


def _patch_transport(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("dataos.llm.openai_compat.httpx.Client", factory)


def test_openai_compat_client_extracts_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = request.content.decode("utf-8")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                'preface {"steps": [{"operator_name": "dedup", '
                                '"params": {}}]} trailer'
                            )
                        }
                    }
                ]
            },
        )

    _patch_transport(monkeypatch, httpx.MockTransport(handler))

    client = OpenAICompatLLMClient(
        api_key="k", base_url="https://example.com", model="demo"
    )
    payload = client.generate_json("system text", "user text")

    assert payload == {"steps": [{"operator_name": "dedup", "params": {}}]}
    assert captured["url"] == "https://example.com/chat/completions"
    assert captured["auth"] == "Bearer k"
    assert "Return exactly one JSON object" in str(captured["body"])
    assert "demo" in str(captured["body"])


def test_openai_compat_client_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_transport(monkeypatch, httpx.MockTransport(handler))

    client = OpenAICompatLLMClient(api_key="k", base_url="https://example.com")
    with pytest.raises(RuntimeError, match="500"):
        client.generate_json("s", "u")


def test_build_client_from_env_prefers_dashscope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key")
    monkeypatch.setenv("DASHSCOPE_MODEL", "qwen-max")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.delenv("STARTEAM_API_KEY", raising=False)
    monkeypatch.delenv("STARTEAM_BASE_URL", raising=False)
    monkeypatch.delenv("STARTEAM_MODEL", raising=False)

    client = _build_client_from_env()

    assert isinstance(client, OpenAICompatLLMClient)
    assert client.api_key == "ds-key"
    assert client.base_url == DASHSCOPE_BASE_URL
    assert client.model == "qwen-max"
    assert DEFAULT_MODEL == "deepseek-v4-flash"


def test_build_client_from_env_falls_back_to_starteam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_MODEL", raising=False)
    monkeypatch.setenv("STARTEAM_API_KEY", "st-key")
    monkeypatch.setenv("STARTEAM_BASE_URL", "https://legacy.example.com")
    monkeypatch.setenv("STARTEAM_MODEL", "glm-5")

    client = _build_client_from_env()

    assert client.api_key == "st-key"
    assert client.base_url == "https://legacy.example.com"
    assert client.model == "glm-5"


def test_build_client_from_env_uses_defaults_with_only_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-only")
    for name in (
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_MODEL",
        "STARTEAM_API_KEY",
        "STARTEAM_BASE_URL",
        "STARTEAM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    client = _build_client_from_env()

    assert client.base_url == DASHSCOPE_BASE_URL
    assert client.model == DEFAULT_MODEL


def test_build_client_from_env_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_MODEL",
        "STARTEAM_API_KEY",
        "STARTEAM_BASE_URL",
        "STARTEAM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        _build_client_from_env()
