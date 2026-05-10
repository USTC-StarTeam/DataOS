# DataOS

[中文说明 / Chinese README](README.zh-CN.md)

DataOS is a local-first, single-experiment runtime for SFT data workflows.
It keeps the LLM boundary narrow: the LLM plans data-operator steps and reviews
the finished experiment, while local Python code owns execution, training,
evaluation, and artifact persistence.

The current architecture is documented in [docs/dataos_overview.md](docs/dataos_overview.md).

## Scope

- Hugging Face `Dataset` / `DatasetDict` `save_to_disk` directories are the
  internal dataset format.
- `run_task()` executes one experiment inside a local sandbox.
- Data operators are deterministic and registered locally.
- `train_llamafactory` and `eval_model` are fixed runtime stages, not planned
  operators.
- Evaluation keeps lm-eval backend compatibility plus a small natural-language
  task helper.
- There is no research layer, daemon loop, project queue, or persistent agent
  control plane.

## Install

```bash
uv sync --extra dev
```

Optional runtime stacks:

```bash
uv sync --extra train
uv sync --extra eval
```

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

## Validation

```bash
uv run ruff check .
```
