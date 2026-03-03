# DataOS

## Python Agent (PR1 Skeleton)

### uv Setup

```bash
uv sync --extra dev
```

### Run

```bash
uv run python -m py_agent.main --help
uv run python -m py_agent.main --version
uv run python -m py_agent.main -p --mode text --model glm-5 "hello"
```

Set provider env vars before print mode:

```bash
export STARTEAM_BASE_URL="https://your-openai-compatible-endpoint"
export STAETEAM_API_KEY="your-api-key"
```

### Validate

```bash
uv run ruff check .
uv run pytest
```
