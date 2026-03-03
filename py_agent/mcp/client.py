"""Minimal stdio MCP client."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 15.0


class McpClientError(RuntimeError):
    """Raised for MCP client failures."""


@dataclass(frozen=True)
class McpToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]


class StdioMcpClient:
    def __init__(
        self,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.server_name = server_name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        self._proc: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._incoming: Queue[dict[str, Any]] = Queue()
        self._request_id = 0

    def start(self) -> None:
        if self._proc is not None:
            return

        try:
            proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.cwd,
                env={**self.env} if self.env else None,
            )
        except Exception as exc:
            raise McpClientError(f"[{self.server_name}] failed to start MCP server: {exc}") from exc

        self._proc = proc
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "py-agent", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools")
        if not isinstance(tools, list):
            raise McpClientError(f"[{self.server_name}] invalid tools/list result")
        return [t for t in tools if isinstance(t, dict)]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        result = self._request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )
        return _extract_text_from_call_result(result)

    def close(self) -> None:
        if self._proc is None:
            return

        proc = self._proc
        self._proc = None
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

    def _reader_loop(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                self._incoming.put(item)

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._write(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = self._next_request_id()
        self._write(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params,
            }
        )
        deadline = time.time() + DEFAULT_TIMEOUT_SECONDS
        while time.time() < deadline:
            if self._proc is None:
                raise McpClientError(f"[{self.server_name}] MCP server is not running")
            if self._proc.poll() is not None:
                stderr_text = ""
                try:
                    if self._proc.stderr:
                        stderr_text = self._proc.stderr.read().strip()
                except Exception:
                    stderr_text = ""
                extra = f", stderr={stderr_text}" if stderr_text else ""
                raise McpClientError(f"[{self.server_name}] MCP server exited unexpectedly{extra}")
            try:
                msg = self._incoming.get(timeout=0.25)
            except Empty:
                continue
            if msg.get("id") != req_id:
                continue
            if "error" in msg:
                err = msg.get("error") or {}
                message = err.get("message") if isinstance(err, dict) else str(err)
                raise McpClientError(f"[{self.server_name}] MCP request failed: {message}")
            result = msg.get("result")
            if not isinstance(result, dict):
                raise McpClientError(f"[{self.server_name}] MCP response missing result")
            return result
        raise McpClientError(f"[{self.server_name}] MCP request timed out: {method}")

    def _write(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise McpClientError(f"[{self.server_name}] MCP server is not running")
        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            raise McpClientError(f"[{self.server_name}] failed to write MCP request: {exc}") from exc

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id


def _extract_text_from_call_result(result: dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)
    if isinstance(content, str):
        return content
    return json.dumps(result, ensure_ascii=False)
