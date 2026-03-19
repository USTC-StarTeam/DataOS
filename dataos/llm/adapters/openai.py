"""OpenAI-compatible LLM adapter for the DataOS unified interface."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable
from urllib import error, request

from dataos.llm.base import GenerateRequest, GenerateResponse, LLMError, LLMMessage, TokenUsage

BASE_URL_ENV = "OPENAI_BASE_URL"
API_KEY_ENV = "OPENAI_API_KEY"
LEGACY_BASE_URL_ENVS = ("STARTEAM_BASE_URL",)
LEGACY_API_KEY_ENVS = ("STARTEAM_API_KEY", "STAETEAM_API_KEY")

Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def _default_transport(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout_seconds: float
) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        details = ""
        try:
            details = exc.read().decode("utf-8", errors="replace")
        except Exception:
            details = ""
        suffix = f" body={details}" if details else ""
        raise LLMError(f"request failed: HTTP {exc.code}{suffix}") from exc
    except Exception as exc:  # pragma: no cover - network failures are environment-dependent.
        raise LLMError(f"request failed: {exc}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError("invalid JSON response") from exc


class OpenAICompatibleLLM:
    """Minimal adapter for providers that expose the OpenAI chat completions API."""

    provider_name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: Transport | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        resolved_base_url = base_url or _first_env(BASE_URL_ENV, *LEGACY_BASE_URL_ENVS)
        if not resolved_base_url:
            raise LLMError(
                f"missing base_url; provide it explicitly or set {BASE_URL_ENV}"
            )

        self.base_url = resolved_base_url.rstrip("/")
        self.api_key = (
            api_key if api_key is not None else _first_env(API_KEY_ENV, *LEGACY_API_KEY_ENVS)
        )
        self.transport = transport or _default_transport
        self.timeout_seconds = timeout_seconds

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        started_at = time.perf_counter()
        url = _build_chat_completions_url(self.base_url)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [_serialize_message(message) for message in request.messages],
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop
        payload.update(request.extra)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = self.transport(url, payload, headers, self.timeout_seconds)
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        try:
            choice = data["choices"][0]
            message = choice["message"]
        except Exception as exc:
            raise LLMError("unexpected response format") from exc

        text = _flatten_message_content(message.get("content"))
        if not text:
            raise LLMError("empty response content")

        usage = _parse_usage(data.get("usage"))
        metadata = {
            "base_url": self.base_url,
            "response_id": data.get("id"),
            "system_fingerprint": data.get("system_fingerprint"),
        }
        if request.metadata:
            metadata["request_metadata"] = dict(request.metadata)

        resolved_model = data.get("model")
        if not isinstance(resolved_model, str) or not resolved_model.strip():
            resolved_model = request.model

        return GenerateResponse(
            provider=self.provider_name,
            model=resolved_model,
            text=text,
            usage=usage,
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
            metadata=metadata,
            raw=data,
        )


def _first_env(*keys: str) -> str:
    for key in keys:
        value = os.getenv(key, "")
        if value:
            return value
    return ""


def _build_chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _serialize_message(message: LLMMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        payload["name"] = message.name
    return payload


def _flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                parts.append(item["text"])
        return "\n".join(parts).strip()

    return ""


def _parse_usage(raw_usage: Any) -> TokenUsage | None:
    if not isinstance(raw_usage, dict):
        return None

    usage = TokenUsage(
        prompt_tokens=_as_int(raw_usage.get("prompt_tokens")),
        completion_tokens=_as_int(raw_usage.get("completion_tokens")),
        total_tokens=_as_int(raw_usage.get("total_tokens")),
    )
    if (
        usage.prompt_tokens is None
        and usage.completion_tokens is None
        and usage.total_tokens is None
    ):
        return None
    return usage


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


__all__ = [
    "API_KEY_ENV",
    "BASE_URL_ENV",
    "LEGACY_API_KEY_ENVS",
    "LEGACY_BASE_URL_ENVS",
    "OpenAICompatibleLLM",
]
