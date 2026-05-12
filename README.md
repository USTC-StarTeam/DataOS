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

Sync the project environment:

```bash
uv sync --extra dev
```

Optional runtime stacks:

```bash
uv sync --extra train
uv sync --extra eval
```

## CLI

```bash
uv run dataos operators list

uv run dataos train \
  --task dataos/templates/sft_v1.json \
  --dataset tatsu-lab/alpaca_eval,openai/gsm8k \
  --workspace workspace
```

If you want planning and review to use DashScope, export the runtime variables
before `uv run`:

```bash
export DASHSCOPE_BASE_URL="your-base-url"
export DASHSCOPE_API_KEY="your-api-key"
export DASHSCOPE_MODEL="your-model"
```

## Validation

```bash
uv run ruff check .
```
