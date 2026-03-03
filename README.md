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
```

### Validate

```bash
ruff check .
pytest
```
