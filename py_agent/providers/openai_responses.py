"""OpenAI-compatible provider (STARTEAM endpoint)."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Callable
from urllib import error, request

from py_agent.debug_log import DebugLogger
from py_agent.providers.message_transform import transform_chat_messages
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
    except error.HTTPError as exc:
        details = ""
        try:
            details = exc.read().decode("utf-8", errors="replace")
        except Exception:
            details = ""
        payload_preview = json.dumps(payload, ensure_ascii=False)[:4000]
        suffix = f" body={details}" if details else ""
        raise ProviderError(f"request failed: HTTP {exc.code}{suffix} payload={payload_preview}") from exc
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
        debug_logger: DebugLogger | None = None,
        debug_level: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv(BASE_URL_ENV, "")).rstrip("/")
        self.api_key = api_key or os.getenv(API_KEY_ENV, "")
        self.transport = transport or _default_transport
        self.debug_logger = debug_logger
        self.debug_level = debug_level

        if not self.base_url:
            raise ProviderError(f"missing environment variable: {BASE_URL_ENV}")
        if not self.api_key:
            raise ProviderError(f"missing environment variable: {API_KEY_ENV}")

    def chat(self, messages: list[dict[str, Any]], model: str, tools: list[dict[str, Any]] | None = None) -> ChatResult:
        url = _build_chat_completions_url(self.base_url)
        transformed_messages = transform_chat_messages(messages)
        payload = {
            "model": model,
            "messages": transformed_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.debug_logger and self.debug_level == "high":
            redacted_headers = dict(headers)
            redacted_headers["Authorization"] = "Bearer ***"
            self.debug_logger.llm(
                f"request url={url} payload={json.dumps(payload, ensure_ascii=False)} "
                f"headers={json.dumps(redacted_headers, ensure_ascii=False)}"
            )
        try:
            data = self.transport(url, payload, headers)
        except ProviderError as exc:
            if _is_tool_role_pairing_error(exc):
                compat_messages = _compat_messages_without_tool_history(transformed_messages)
                compat_payload = dict(payload)
                compat_payload["messages"] = compat_messages
                if self.debug_logger and self.debug_level == "high":
                    self.debug_logger.llm("retrying with compatibility message transform (drop historical tool roles)")
                data = self.transport(url, compat_payload, headers)
            else:
                raise
        if self.debug_logger and self.debug_level == "high":
            self.debug_logger.llm(f"response body={json.dumps(data, ensure_ascii=False)}")

        try:
            message = data["choices"][0]["message"]
        except Exception as exc:
            raise ProviderError("unexpected response format") from exc

        text = _flatten_message_content(message.get("content"))
        tool_calls_raw = message.get("tool_calls")
        tool_calls = _parse_tool_calls(tool_calls_raw)
        if not text and not tool_calls:
            raise ProviderError("empty response content")
        assistant_message = {"role": "assistant", "tool_calls": tool_calls_raw}
        if not tool_calls_raw:
            assistant_message["content"] = message.get("content")
        return ChatResult(model=model, text=text, tool_calls=tool_calls, assistant_message=assistant_message)


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


def _is_tool_role_pairing_error(exc: ProviderError) -> bool:
    text = str(exc).lower()
    return 'messages with role "tool"' in text or ("messages with role" in text and "tool" in text)


def _compat_messages_without_tool_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            continue
        if role == "assistant" and msg.get("tool_calls") is not None:
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                out.append({"role": "assistant", "content": content})
            continue
        out.append(msg)
    return out
