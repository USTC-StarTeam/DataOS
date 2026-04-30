from __future__ import annotations

import httpx
import pytest

from dataos.llm.starteam import StarteamLLMClient
from dataos.runtime.runner import _build_client_from_env


_REAL_HTTPX_CLIENT = httpx.Client


def _patch_transport(monkeypatch: pytest.MonkeyPatch, transport: httpx.MockTransport) -> None:
    def factory(*args, **kwargs):
        kwargs["transport"] = transport
        return _REAL_HTTPX_CLIENT(*args, **kwargs)

    monkeypatch.setattr("dataos.llm.starteam.httpx.Client", factory)


def test_starteam_llm_client_extracts_json(monkeypatch: pytest.MonkeyPatch) -> None:
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
                                'preface {"steps": [{"operator_name": "clean_data", '
                                '"params": {}}]} trailer'
                            )
                        }
                    }
                ]
            },
        )

    _patch_transport(monkeypatch, httpx.MockTransport(handler))

    client = StarteamLLMClient(api_key="k", base_url="https://example.com", model="demo")
    payload = client.generate_json("system text", "user text")

    assert payload == {"steps": [{"operator_name": "clean_data", "params": {}}]}
    assert captured["url"] == "https://example.com/chat/completions"
    assert captured["auth"] == "Bearer k"
    assert "Return exactly one JSON object" in str(captured["body"])
    assert "demo" in str(captured["body"])


def test_starteam_llm_client_raises_on_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_transport(monkeypatch, httpx.MockTransport(handler))

    client = StarteamLLMClient(api_key="k", base_url="https://example.com")
    with pytest.raises(RuntimeError, match="500"):
        client.generate_json("s", "u")


def test_build_client_from_env_returns_starteam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STARTEAM_API_KEY", "k")
    monkeypatch.setenv("STARTEAM_BASE_URL", "https://example.com")
    monkeypatch.setenv("STARTEAM_MODEL", "demo-model")

    client = _build_client_from_env()

    assert isinstance(client, StarteamLLMClient)
    assert client.model == "demo-model"


def test_build_client_from_env_requires_starteam_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STARTEAM_API_KEY", raising=False)
    monkeypatch.delenv("STARTEAM_BASE_URL", raising=False)
    monkeypatch.delenv("STARTEAM_MODEL", raising=False)

    with pytest.raises(ValueError, match="STARTEAM_API_KEY and STARTEAM_BASE_URL"):
        _build_client_from_env()
