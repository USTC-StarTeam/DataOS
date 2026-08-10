"""claude — the ``claude`` CLI (Claude Code) in headless stream-json mode.

Stream shape (also used by qwen, a Claude-Code-derived CLI):
  system     -> init: session_id, model, tools
  assistant  -> {message:{content:[{type:text}|{type:tool_use}]}}
  user       -> {message:{content:[{type:tool_result}]}}
  result     -> {is_error, result:<final text>, session_id, usage, total_cost_usd, stop_reason}
"""

from __future__ import annotations

from typing import Iterable

from ..types import AgentConfig, Event, Result, Step
from .base import CliSpec, register

_STOP_TO_STATUS = {"max_tokens": "max_steps", "max_turns": "max_steps"}


def _blocks_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


class ClaudeStyleSpec(CliSpec):
    """Maps the Claude-Code stream-json format into neutral Events + a Result.
    Shared by claude and qwen; each only differs in ``bin`` + ``build_argv``."""

    def parse_line(self, raw: dict, acc: Result) -> Iterable[Event]:
        t = raw.get("type")
        if t == "system":
            acc.session_id = raw.get("session_id") or acc.session_id
            return
        if t == "assistant":
            for block in (raw.get("message") or {}).get("content", []):
                bt = block.get("type")
                if bt == "text":
                    txt = block.get("text", "")
                    yield Event(type="text", text=txt)
                elif bt == "tool_use":
                    name, args = block.get("name", ""), block.get("input") or {}
                    acc.steps.append(Step(tool=name, args=args))
                    yield Event(type="tool_use", name=name, args=args)
            return
        if t == "user":
            for block in (raw.get("message") or {}).get("content", []):
                if block.get("type") == "tool_result":
                    text = _blocks_text(block.get("content"))
                    if acc.steps:
                        acc.steps[-1].result = text
                    yield Event(type="tool_result", data=text)
            return
        if t == "result":
            if raw.get("result") is not None:
                acc.text = raw["result"]            # authoritative final text
            u = raw.get("usage") or {}
            acc.usage.add(input=u.get("input_tokens", 0), output=u.get("output_tokens", 0))
            acc.usage.cost_usd += raw.get("total_cost_usd") or 0.0
            acc.session_id = raw.get("session_id") or acc.session_id
            if raw.get("is_error"):
                if "max_turns" in str(raw.get("subtype") or ""):
                    # benign, RESUMABLE boundary: the caller drives the session in
                    # max-turns-sized chunks — not an error.
                    acc.status = "max_steps"
                else:
                    acc.status = "error"
                    acc.error = raw.get("result") or raw.get("subtype") or "error"
            else:
                acc.status = _STOP_TO_STATUS.get(raw.get("stop_reason"), "done")
            return


class ClaudeSpec(ClaudeStyleSpec):
    name = "claude"
    bin = "claude"
    # the Anthropic-reachable egress. Previously came from ``~/.claude/settings.json``'s
    # env block — declared here so it survives an isolated config root (config.home).
    proxy = "http://118.89.136.118:31094"

    def build_argv(self, config: AgentConfig, prompt: str) -> list[str]:
        argv = ["-p", prompt, "--output-format", "stream-json", "--verbose"]
        if config.model:
            argv += ["--model", config.model]
        if config.effort:
            argv += ["--effort", config.effort]   # low | medium | high | xhigh | max
        if config.instructions:
            argv += ["--append-system-prompt", config.instructions]
        if config.max_steps:
            argv += ["--max-turns", str(config.max_steps)]
        if isinstance(config.tools, list):
            argv += ["--allowed-tools", " ".join(config.tools)]
        if config.tools and config.permission == "auto":
            argv += ["--dangerously-skip-permissions"]   # tools=False -> omit -> tools can't run (pure reasoning)
        if config.extra.get("resume"):
            argv += ["--resume", config.extra["resume"]]
        argv += list(config.extra.get("args", []))
        return argv


register(ClaudeSpec())
