"""qwen — the ``qwen`` CLI (Qwen Code). Same stream-json shape as claude, so it
reuses ClaudeStyleSpec; only the argv differs. Qwen has no --append-system-prompt,
so instructions are prepended to the prompt (graceful degradation)."""

from __future__ import annotations

from ..types import AgentConfig
from .base import register
from .claude import ClaudeStyleSpec


class QwenSpec(ClaudeStyleSpec):
    name = "qwen"
    bin = "qwen"

    def build_argv(self, config: AgentConfig, prompt: str) -> list[str]:
        # no system-prompt flag -> prepend. Unconditional (unlike codex/kimi): this
        # provider never passes a resume id to the CLI, so every call is a fresh session.
        if config.instructions:
            prompt = f"{config.instructions}\n\n{prompt}"
        argv = ["-p", prompt, "--output-format", "stream-json"]
        if config.model:
            argv += ["-m", config.model]
        if config.tools and config.permission == "auto":
            argv += ["--yolo"]                        # auto-approve tools
        argv += list(config.extra.get("args", []))
        return argv


register(QwenSpec())
