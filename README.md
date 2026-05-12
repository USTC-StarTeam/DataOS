# DataOS

[中文说明 / Chinese README](README.zh-CN.md)

DataOS is a local-first, single-experiment runtime for SFT data workflows.
It keeps the LLM boundary mostly narrow: the planning agent still handles plan
and review, while local Python code owns execution, training, evaluation, and
artifact persistence. Some evaluator and generator operators may optionally use
an OpenAI-compatible model when the task explicitly requires scoring or
augmentation.

The current architecture is documented in [docs/dataos_overview.md](docs/dataos_overview.md).
The operator catalog is documented in [docs/operators_zh.md](docs/operators_zh.md).

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

If `uv` is installed under `~/.local/bin`, add it to your shell first:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then sync the project environment:

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
    datasets=[
        "alpaca-eval",
        "gsm8k",
    ],
    workspace_dir=Path("workspace"),
)

print(result.experiment.status)
print(result.review)
```

## CLI

```bash
uv run dataos operators list

uv run dataos train \
  --task dataos/templates/sft_v1.json \
  --dataset alpaca-eval,gsm8k \
  --workspace workspace
```

## Validation

```bash
uv run ruff check .
```
