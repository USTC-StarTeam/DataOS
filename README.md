# DataOS

DataOS V0.1 is a local-first, agentic data system for SFT workflows. The LLM is only used for planning and review, while a fixed runtime executes registered operators in sequence inside an isolated sandbox.

## Scope

- Internal dataset flow is standardized on Hugging Face `Dataset` and `DatasetDict` directories created by `save_to_disk`.
- `DatasetArtifact` only wraps Hugging Face datasets directories.
- Operators execute in a sandbox and emit typed artifacts.
- Experiments write full reports and review outputs to disk.

## Install

```bash
uv sync --extra dev
```

## Package Layout

- `dataos/core`: artifacts, sandbox, experiment, registry, store, task, agent
- `dataos/operators`: built-in clean, synthesize, train, eval operators
- `dataos/llm`: minimal structured JSON client interface
- `dataos/prompts`: prompt rules and runtime validators
- `dataos/runtime`: task runner
- `dataos/templates`: example task template

## Quick Start

```python
from pathlib import Path

from dataos.runtime import run_task

result = run_task(
    task_template=Path("dataos/templates/sft_v1.json"),
    dataset_path=Path("/path/to/hf_dataset"),
    workspace_dir=Path("workspace"),
)

print(result.experiment.status)
print(result.review)
```

If you do not pass a custom `llm_client`, set:

- `DATAOS_LLM_BASE_URL`
- `DATAOS_LLM_API_KEY`
- `DATAOS_LLM_MODEL`

## Validation

```bash
uv run pytest
```
