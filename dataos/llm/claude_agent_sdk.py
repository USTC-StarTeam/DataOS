from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataos.llm.client import LLMClient
from dataos.llm.openai_bridge import OpenAICompatAnthropicBridge, StarteamConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Claude Agent SDK response")
    return json.loads(match.group(0))


@dataclass(slots=True)
class ClaudeAgentSDKClient(LLMClient):
    api_key: str
    base_url: str
    model: str = "glm-5"
    cli_path: Path | str | None = None
    cwd: Path | str | None = None
    max_turns: int = 2

    def __post_init__(self) -> None:
        self.cwd = Path(self.cwd) if self.cwd is not None else _project_root()
        self.cli_path = self._resolve_cli_path(self.cli_path)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        del temperature  # Claude Agent SDK query path does not expose temperature here.
        prompt = (
            f"{system_prompt}\n\n"
            "Return exactly one JSON object and nothing else.\n\n"
            f"{user_prompt}"
        )
        text = asyncio.run(self._run_prompt(prompt))
        return _extract_json_object(text)

    async def _run_prompt(self, prompt: str) -> str:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, query

        bridge = OpenAICompatAnthropicBridge(
            StarteamConfig(api_key=self.api_key, base_url=self.base_url, model=self.model)
        )
        bridge.start()
        rendered: list[str] = []
        try:
            options = ClaudeAgentOptions(
                cli_path=self.cli_path,
                cwd=self.cwd,
                permission_mode="bypassPermissions",
                allowed_tools=[],
                max_turns=self.max_turns,
                env={
                    "ANTHROPIC_API_KEY": self.api_key,
                    "ANTHROPIC_BASE_URL": bridge.base_url,
                },
            )
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    parts = [getattr(block, "text", None) for block in message.content]
                    text = "\n".join(part for part in parts if part)
                    if text:
                        rendered.append(text)
        finally:
            bridge.stop()
        if not rendered:
            raise RuntimeError("Claude Agent SDK returned no assistant text")
        return "\n".join(rendered).strip()

    @staticmethod
    def _resolve_cli_path(cli_path: Path | str | None) -> Path:
        if cli_path is not None:
            return Path(cli_path)
        env_value = os.environ.get("DATAOS_CLAUDE_CLI_PATH")
        if env_value:
            return Path(env_value)
        bundled = _project_root() / "bin" / "claude-node22"
        if bundled.exists():
            return bundled
        discovered = shutil.which("claude")
        if discovered:
            return Path(discovered)
        raise RuntimeError(
            "Claude Agent SDK client requires a Claude CLI binary. Set DATAOS_CLAUDE_CLI_PATH or install `claude` in PATH."
        )
