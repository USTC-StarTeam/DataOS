"""OpenAI-compatible provider (STARTEAM endpoint)."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Callable
from urllib import request

from py_agent.providers.base import ChatResult, Provider, ProviderError, ToolCall

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

    def chat(self, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]] | None = None) -> ChatResult:
        url = _build_chat_completions_url(self.base_url)
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = self.transport(url, payload, headers)

        try:
            message = data["choices"][0]["message"]
        except Exception as exc:
            raise ProviderError("unexpected response format") from exc

        text = _flatten_message_content(message.get("content"))
        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        if not text and not tool_calls:
            raise ProviderError("empty response content")
        return ChatResult(model=model, text=text, tool_calls=tool_calls)


def _build_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


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


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    if not isinstance(raw, list):
        return []

    parsed: list[ToolCall] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        fn = item.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        raw_args = fn.get("arguments", "{}")
        try:
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else {}
        except json.JSONDecodeError:
            arguments = {}
        call_id = item.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"toolcall-{idx}-{uuid.uuid4().hex[:8]}"
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return parsed
