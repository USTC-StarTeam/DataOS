"""kimi — the ``kimi`` CLI (Kimi Code) headless: ``-p <prompt> --output-format stream-json``.

Stream shape (role-based message objects, no result/usage line):
  {role:"assistant", content:"<text>" | [{type:"think"|"text"}],
   tool_calls:[{function:{name,arguments(JSON string)}}]}
  {role:"tool", content:<tool output>, tool_call_id:...}
  {role:"meta", type:"session.resume_hint", session_id:"session_<uuid>"}  → session id
Resume via ``-S <session_id>``. No token usage is emitted (usage stays 0 — documented
degradation). Compatible with the new (0.31.x, string content + meta hint) and old
(--print era, content blocks + stderr hint) CLI shapes.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from ..types import AgentConfig, Event, Result, Step
from .base import CliSpec, register

_SESSION_RE = re.compile(r"kimi -r ([0-9a-fA-F-]{8,})")


class KimiSpec(CliSpec):
    name = "kimi"
    bin = "kimi"

    def build_argv(self, config: AgentConfig, prompt: str) -> list[str]:
        resume = config.extra.get("resume")
        if config.instructions and not resume:        # no system-prompt flag -> prepend;
            prompt = f"{config.instructions}\n\n{prompt}"   # a resumed session already has it
        argv = ["-p", prompt, "--output-format", "stream-json"]
        if config.model:
            argv += ["-m", config.model]
        # no workspace flag in the new CLI: the subprocess cwd IS the workspace
        # (run_cli spawns with cwd=config.workspace)
        if resume:
            argv += ["-S", resume]
        argv += list(config.extra.get("args", []))
        return argv

    def parse_line(self, raw: dict, acc: Result) -> Iterable[Event]:
        role = raw.get("role")
        if role == "meta":                             # session resume hint (new CLI)
            if raw.get("type") == "session.resume_hint" and raw.get("session_id"):
                acc.session_id = raw["session_id"]
            return
        if role == "tool":                             # tool output line
            text = str(raw.get("content") or "")
            if acc.steps:
                acc.steps[-1].result = text
            yield Event(type="tool_result", data=text)
            return
        if role != "assistant":
            return
        for tc in raw.get("tool_calls") or []:         # OpenAI-style tool calls
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            acc.steps.append(Step(tool=name, args=args))
            yield Event(type="tool_use", name=name, args=args)
        content = raw.get("content")
        if isinstance(content, str):                   # new CLI: plain text content
            if content:
                acc.text = content
                yield Event(type="text", text=content)
            return
        texts = []
        for block in content or []:                    # old CLI: content blocks
            if block.get("type") == "text":
                txt = block.get("text", "")
                texts.append(txt)
                yield Event(type="text", text=txt)
            # think blocks are internal — skipped
        if texts:
            acc.text = "".join(texts)                 # last assistant message wins

    def finalize(self, acc: Result, stderr: str) -> None:
        if not acc.session_id:
            m = _SESSION_RE.search(stderr)
            if m:
                acc.session_id = m.group(1)


register(KimiSpec())
