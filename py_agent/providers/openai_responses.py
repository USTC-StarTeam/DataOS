"""OpenAI-compatible provider (STARTEAM endpoint)."""

from __future__ import annotations

import json
import os
from typing import Any, Callable
from urllib import request

from py_agent.providers.base import CompletionResult, Provider, ProviderError

BASE_URL_ENV = "STARTEAM_BASE_URL"
API_KEY_ENV = "STAETEAM_API_KEY"
DEFAULT_MODEL = "glm-5"

Transport = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


def _default_transport(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except Exception as exc:  # pragma: no cover - exercised via integration tests
        raise ProviderError(f"request failed: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError("invalid JSON response") from exc


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv(BASE_URL_ENV, "")).rstrip("/")
        self.api_key = api_key or os.getenv(API_KEY_ENV, "")
        self.transport = transport or _default_transport

        if not self.base_url:
            raise ProviderError(f"missing environment variable: {BASE_URL_ENV}")
        if not self.api_key:
            raise ProviderError(f"missing environment variable: {API_KEY_ENV}")

    def complete(self, prompt: str, model: str) -> CompletionResult:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = self.transport(url, payload, headers)

        try:
            choice = data["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ProviderError("unexpected response format") from exc

        text = _flatten_message_content(choice)
        if not text:
            raise ProviderError("empty response content")
        return CompletionResult(model=model, text=text)


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()

    return ""
