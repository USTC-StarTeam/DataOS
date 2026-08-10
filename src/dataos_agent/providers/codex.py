"""codex — the ``codex`` CLI (OpenAI Codex) in headless JSONL mode (``codex exec --json``).

Event shape:
  {"type":"thread.started","thread_id":...}
  {"type":"item.started|item.completed","item":{"type":"agent_message"|"command_execution"|"file_change", ...}}
  {"type":"turn.completed","usage":{input_tokens, output_tokens, ...}}
Resume: ``codex exec resume <thread_id> <prompt>``. Effort via ``-c model_reasoning_effort=<level>``.
"""

from __future__ import annotations

from typing import Iterable

from ..types import AgentConfig, Event, Result, Step
from .base import CliSpec, register

# Codex silently truncates its global context file when it exceeds the project-doc
# budget ("project doc exceeds remaining budget; truncating" — no error, no exit code),
# which would eat the tail of a large AGENTS.md. Set the budget high enough to never hit it.
_PROJECT_DOC_MAX_BYTES = 262144


class CodexSpec(CliSpec):
    name = "codex"
    bin = "codex"
    proxy = "http://192.168.1.99:8401"                # the OpenAI-reachable egress

    def build_argv(self, config: AgentConfig, prompt: str) -> list[str]:
        resume = config.extra.get("resume")
        if config.instructions and not resume:        # no system-prompt flag -> prepend;
            prompt = f"{config.instructions}\n\n{prompt}"   # a resumed thread already has it
        argv = ["exec"]
        if resume:
            argv += ["resume", resume]
        argv += ["--json", "--skip-git-repo-check",
                 "-c", f"project_doc_max_bytes={_PROJECT_DOC_MAX_BYTES}"]
        if config.model:
            argv += ["-m", config.model]
        if config.effort:
            argv += ["-c", f"model_reasoning_effort={config.effort}"]
        if config.workspace and not resume:
            # `exec resume` rejects -C; the resumed thread keeps its original cwd,
            # and run_cli already spawns the process with cwd=config.workspace
            argv += ["-C", config.workspace]
        if config.tools and config.permission == "auto":
            argv += ["--dangerously-bypass-approvals-and-sandbox"]
        elif not config.tools:
            argv += ["-s", "read-only"]
        argv += list(config.extra.get("args", []))
        argv.append(prompt)
        return argv

    def parse_line(self, raw: dict, acc: Result) -> Iterable[Event]:
        t = raw.get("type")
        if t == "thread.started":
            acc.session_id = raw.get("thread_id") or acc.session_id
            return
        if t == "item.completed":
            item = raw.get("item") or {}
            it = item.get("type")
            if it == "agent_message":
                text = item.get("text", "")
                acc.text = text                          # last agent message wins
                yield Event(type="text", text=text)
            elif it in ("command_execution", "file_change"):
                name = "Bash" if it == "command_execution" else "Edit"
                args = (
                    {"command": item.get("command")}
                    if it == "command_execution"
                    else {"changes": item.get("changes")}
                )
                result = str(item.get("aggregated_output") or item.get("status") or "")
                acc.steps.append(Step(tool=name, args=args, result=result))
                yield Event(type="tool_use", name=name, args=args)
                yield Event(type="tool_result", data=result)
            # reasoning / plan items are internal — skipped
            return
        if t == "turn.completed":
            u = raw.get("usage") or {}
            acc.usage.add(input=u.get("input_tokens", 0), output=u.get("output_tokens", 0))
            return
        # {"type":"error","message":"Reconnecting..."} lines are codex's own retry
        # chatter — non-terminal, the process outcome decides status


register(CodexSpec())
