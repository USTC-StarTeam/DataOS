# DataOS

[中文说明 / Chinese README](README.zh-CN.md)

DataOS is a local-first, single-experiment runtime for SFT data workflows.
It keeps the LLM boundary mostly narrow: the planning agent still handles plan
and review, while local Python code owns execution, training, evaluation, and
artifact persistence. Some evaluator and generator operators may optionally use
an OpenAI-compatible model when the task explicitly requires scoring or
augmentation.

The current architecture is documented in
[docs/overview/dataos_overview.md](docs/overview/dataos_overview.md).
The operator catalog is documented in
[docs/architecture/operators_zh.md](docs/architecture/operators_zh.md).

## Scope

- Hugging Face `Dataset` / `DatasetDict` `save_to_disk` directories are the
  internal dataset format.
- `run_task()` executes one experiment inside a local experiment directory.
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

## Repository Layout

```text
src/dataos/      DataOS package: agent loop, core objects, runtime, integrations
configs/         Runtime and experiment configs
resources/data/  HF datasets and benchmark data resources
resources/code/  Benchmark/harness code resources used by experiments
docs/            Architecture notes and runbooks
scripts/         Thin command-line helpers
tests/           Tests grouped by agent/core/runtime/integrations
```

## CLI

```bash
uv run dataos catalog

uv run dataos-auto run \
  --project gsm8k-smoke \
  --target-bench gsm8k \
  --ptb-profile smoke \
  --runtime-config configs/runtimes/h100.json
```

If you want planning / review to use DashScope, export these variables before `uv run`:

```bash
export DASHSCOPE_BASE_URL="your-base-url"
export DASHSCOPE_API_KEY="your-api-key"
export DASHSCOPE_MODEL="your-model"
```

If you also use `evaluate_llm_pair`, configure the evaluator endpoint separately:

```bash
export EVALUATOR_BASE_URL="your-evaluator-base-url"
export EVALUATOR_API_KEY="your-evaluator-api-key"
export EVALUATOR_MODEL="your-evaluator-model"
```

For mixed-dataset runs, the default conversational dataset shape is ShareGPT-style `conversations` rows.

For the experiment directory layout and saved files, see
[docs/architecture/experiment_layout_zh.md](docs/architecture/experiment_layout_zh.md).

## Validation

```bash
uv run ruff check .
```
