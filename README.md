# DataOS

## Agentic DataOS Docs

SFT 数据闭环 MVP 设计文档位于 `doc/dataos/`：

- `doc/dataos/README.md`
- `doc/dataos/t0-baseline-contract.md`
- `doc/dataos/sft-dataos-mvp-requirements.md`
- `doc/dataos/sft-dataos-mvp-design.md`
- `doc/dataos/sft-dataos-mvp-task-breakdown.md`
- `doc/dataos/tasks/`

## Python Agent (MVP In Progress)

### 1) Setup (uv)

```bash
uv sync --extra dev
```

### 2) Provider Environment

```bash
export STARTEAM_BASE_URL="https://your-openai-compatible-endpoint"
export STAETEAM_API_KEY="your-api-key"
```

### 3) Basic CLI Usage

```bash
uv run python -m py_agent.main --help
uv run python -m py_agent.main --version
uv run python -m py_agent.main -p --mode text --model glm-5 "hello"
uv run python -m py_agent.main -p --mode json --model glm-5 "hello"
uv run python -m py_agent.main --mode tui --model glm-5
uv run python -m py_agent.main --debug low --version
uv run python -m py_agent.main --debug high --log-path ./logs/agent-debug.log --version
```

### 4) Session Persistence

Default session file:
- `./.py-agent/sessions/default.jsonl`

Examples:

```bash
# continue default session
uv run python -m py_agent.main -p --continue "continue with prior context"

# use custom session file
uv run python -m py_agent.main -p --session ./my-session.jsonl "store session here"

# continue custom session
uv run python -m py_agent.main -p --session ./my-session.jsonl --continue "next message"

# disable session persistence
uv run python -m py_agent.main -p --no-session "one-shot"

# TUI mode + continue
uv run python -m py_agent.main --mode tui --continue
```

### 4.1) Debug Logging

- Enable with `--debug`
- `--debug low`: only runtime events (`EVENT` lines)
- `--debug high`: full logs (flow + errors + LLM request/response with redacted auth)
- Default log path: `./.py-agent/logs/debug.json`
- Custom path: `--log-path <file>` (must be used with `--debug`)
- Log format: JSON Lines (one JSON object per line)
- Log timestamp timezone: Asia/Shanghai (UTC+08:00)

### 5) MCP Usage

Run with MCP enabled:

```bash
uv run python -m py_agent.main -p --mcp-config ./mcp.json --mcp-server demo "use mcp tools"
uv run python -m py_agent.main --mode tui --mcp-config ./mcp.json --mcp-server demo
```

`mcp.json` example:

```json
{
  "servers": {
    "demo": {
      "command": "python",
      "args": ["tests/fixtures/mcp/mock_server.py"],
      "cwd": "."
    }
  }
}
```

### 6) Skills Usage

Prompt with skill invocation token:

```bash
uv run python -m py_agent.main -p "/skill:writer draft release notes"
```

Skill discovery paths:
- `skills/**/SKILL.md`
- `.codex/skills/**/SKILL.md`

TUI tips:
- Exit with `/exit` or `/quit`
- `Ctrl+C` / `Ctrl+D` exits gracefully

### Current Capabilities

- CLI entrypoint and args (`--help`, `--version`, `-p/--print`, `--mode text|json`, `--model`)
- Session persistence flags (`--no-session`, `--session <path>`, `--continue`)
- OpenAI-compatible provider (tested with `glm-5`)
- Local tools: `read`, `write`, `edit`, `bash`
- Runtime loop: model -> local tool calls -> model (with gateway compatibility fallback)
- MCP integration: stdio client, namespaced tools (`mcp.<server>.<tool>`), explicit server allowlist (`--mcp-server`)
- Skills: discover `SKILL.md` from `skills/` and `.codex/skills/`, invoke via `/skill:<name>`, and inject skill catalog into system prompt

### Validation

Common commands:

```bash
uv run ruff check .
uv run pytest -m "not integration and not task_eval and not mcp_external"
uv run pytest -m integration
uv run pytest -m task_eval
uv run pytest tests/test_integration_mcp_external.py -m mcp_external -q
```

Test layers:

- `not integration and not task_eval and not mcp_external`:
  deterministic unit tests (blocking)
- `integration`:
  deterministic API connectivity and basic program integration (blocking in CI)
- `task_eval`:
  model-capability task evaluations (non-blocking in CI, used as observability signals)
- `mcp_external`:
  manual third-party MCP connectivity test (fixed to `@modelcontextprotocol/server-everything`, not run in CI by default)

Test workspace policy:

- File-producing tests use a fixed workspace root: `tests/.tmp/`
- Each test case gets its own subdirectory under `tests/.tmp`
- Test subdirectories are deleted after each test run
