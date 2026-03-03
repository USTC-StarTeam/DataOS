# DataOS

## Python Agent (MVP In Progress)

### uv Setup

```bash
uv sync --extra dev
```

### Run

```bash
uv run python -m py_agent.main --help
uv run python -m py_agent.main --version
uv run python -m py_agent.main -p --mode text --model glm-5 "hello"
uv run python -m py_agent.main -p --mode json --model glm-5 "hello"
uv run python -m py_agent.main -p --continue "continue with prior context"
uv run python -m py_agent.main -p --session ./my-session.jsonl "store session here"
uv run python -m py_agent.main -p --mcp-config ./mcp.json --mcp-server demo "use mcp tools"
```

Set provider env vars before print mode:

```bash
export STARTEAM_BASE_URL="https://your-openai-compatible-endpoint"
export STAETEAM_API_KEY="your-api-key"
```

MCP config example (`mcp.json`):

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

### Current Capabilities

- CLI entrypoint and args (`--help`, `--version`, `-p/--print`, `--mode text|json`, `--model`)
- Session persistence flags (`--no-session`, `--session <path>`, `--continue`)
- OpenAI-compatible provider (tested with `glm-5`)
- Local tools: `read`, `write`, `edit`, `bash`
- Runtime loop: model -> local tool calls -> model (with gateway compatibility fallback)
- MCP integration: stdio client, namespaced tools (`mcp.<server>.<tool>`), explicit server allowlist (`--mcp-server`)

### Validation

Run all checks:

```bash
uv run ruff check .
uv run pytest -m "not integration and not task_eval"
uv run pytest -m integration
uv run pytest -m task_eval
uv run pytest -m mcp_external
```

Test layers:

- `not integration and not task_eval`:
  deterministic unit tests (blocking)
- `integration`:
  deterministic API connectivity and basic program integration (blocking in CI)
- `task_eval`:
  model-capability task evaluations (non-blocking in CI, used as observability signals)
- `mcp_external`:
  manual third-party MCP connectivity tests (not run in CI by default)

`mcp_external` is fixed to use `@modelcontextprotocol/server-everything` via `npx`:

```bash
uv run pytest tests/test_integration_mcp_external.py -m mcp_external -q
```

Test workspace policy:

- File-producing tests use a fixed workspace root: `tests/.tmp/`
- Each test case gets its own subdirectory under `tests/.tmp`
- Test subdirectories are deleted after each test run
