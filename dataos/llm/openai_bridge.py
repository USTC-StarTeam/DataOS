from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _to_openai_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system_blocks = payload.get("system", [])
    if isinstance(system_blocks, str):
        system_text = system_blocks
    else:
        system_text = "\n".join(
            item.get("text", "")
            for item in system_blocks
            if isinstance(item, dict) and item.get("type") == "text"
        )
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for item in payload.get("messages", []):
        role = item.get("role", "user")
        messages.append({"role": role, "content": _flatten_content(item.get("content"))})
    return messages


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@dataclass(slots=True)
class StarteamConfig:
    api_key: str
    base_url: str
    model: str = "glm-5"


class OpenAICompatAnthropicBridge:
    """Minimal Anthropic Messages API shim backed by an OpenAI-compatible chat API."""

    def __init__(self, config: StarteamConfig) -> None:
        self.config = config
        self.port = _pick_free_port()
        self._server = None
        self._thread: threading.Thread | None = None
        self._app = self._build_app()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _build_app(self) -> Any:
        from starlette.applications import Starlette
        from starlette.routing import Route

        return Starlette(
            routes=[
                Route("/v1/messages", self.handle_messages, methods=["POST"]),
                Route("/v1/models", self.handle_models, methods=["GET"]),
            ]
        )

    def start(self) -> None:
        import uvicorn

        config = uvicorn.Config(self._app, host="127.0.0.1", port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        deadline = time.time() + 10
        while time.time() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            time.sleep(0.05)
        raise RuntimeError("Timed out while starting the local STARTEAM bridge")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

    async def handle_models(self, request: Any) -> Any:
        del request
        from starlette.responses import JSONResponse

        return JSONResponse(
            {
                "data": [
                    {
                        "id": self.config.model,
                        "type": "model",
                        "display_name": self.config.model,
                    }
                ]
            }
        )

    async def handle_messages(self, request: Any) -> Any:
        from starlette.responses import JSONResponse, StreamingResponse
        import httpx

        payload = await request.json()
        remote_payload = {
            "model": self.config.model,
            "messages": _to_openai_messages(payload),
            "stream": False,
        }

        remote_url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=120) as client:
            remote_response = await client.post(remote_url, headers=headers, json=remote_payload)

        if remote_response.status_code >= 400:
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": remote_response.text or f"Upstream error {remote_response.status_code}",
                    },
                },
                status_code=remote_response.status_code,
            )

        data = remote_response.json()
        text = data["choices"][0]["message"]["content"]
        prompt_tokens = int(data.get("usage", {}).get("prompt_tokens", 0))
        completion_tokens = int(data.get("usage", {}).get("completion_tokens", 0))
        requested_model = str(payload.get("model", "claude-sonnet-4-5"))
        message_id = f"msg_{uuid.uuid4().hex}"

        if payload.get("stream", False):

            async def event_stream() -> Any:
                yield _sse_event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": requested_model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                        },
                    },
                )
                yield _sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                yield _sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                )
                yield _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})
                yield _sse_event(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": completion_tokens},
                    },
                )
                yield _sse_event("message_stop", {"type": "message_stop"})

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return JSONResponse(
            {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": requested_model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
            }
        )
