# DataOS

## Python Agent (PR1 Skeleton)

### Conda Setup

```bash
conda env update -n dataos-agent -f environment.yml
conda activate dataos-agent
```

### Run

```bash
python -m py_agent.main --help
python -m py_agent.main --version
python -m py_agent.main -p --mode text --model glm-5 "hello"
```

Set provider env vars before print mode:

```bash
export STARTEAM_BASE_URL="https://your-openai-compatible-endpoint"
export STAETEAM_API_KEY="your-api-key"
```

### Validate

```bash
ruff check .
pytest
```
