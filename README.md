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

If you do not pass a custom `llm_client`, DataOS now expects the agent layer to use the
Claude Agent SDK-backed client. Set:

- `STARTEAM_BASE_URL`
- `STARTEAM_API_KEY`
- optional `STARTEAM_MODEL` defaulting to `glm-5`
- optional `DATAOS_CLAUDE_CLI_PATH` pointing to the Claude CLI binary

DataOS still uses the same fixed runtime and operators. The only change is that
plan/review JSON generation goes through Claude Agent SDK via a local Anthropic-to-OpenAI bridge.

## Validation

```bash
uv run pytest
```
