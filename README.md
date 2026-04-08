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

For real LLaMA-Factory training, install PyTorch for your target CPU/CUDA environment first, then
install the optional train stack:

```bash
uv sync --extra dev --extra train
llamafactory-cli version
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

## LLaMA-Factory Runtime Stages

DataOS now treats training and evaluation as fixed runtime stages rather than LLM-planned
operators. The LLM only plans data operators such as `clean_data` and `synthesize_data`.
After the operator plan finishes, the runtime automatically runs:

- `train_llamafactory`
- `eval_model`

`train_llamafactory` keeps a `stub` mode for fast local tests and also supports a real
`llamafactory` mode driven by `task.train_config`.

Example:

```json
{
  "train_config": {
    "mode": "llamafactory",
    "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
    "template": "qwen",
    "finetuning_type": "lora",
    "extra_args": {
      "cutoff_len": 512,
      "per_device_train_batch_size": 1,
      "gradient_accumulation_steps": 8,
      "learning_rate": 1.0e-4,
      "num_train_epochs": 1.0,
      "logging_steps": 1,
      "save_steps": 10
    }
  }
}
```

In `llamafactory` mode, the runtime training functions:

- loads the latest `DatasetArtifact` from Hugging Face `save_to_disk`
- requires `instruction` and `response`, with optional `input`, `system`, and `history`
- exports an Alpaca-style dataset plus `dataset_info.json` inside the sandbox
- runs `llamafactory-cli train <sandbox config>`
- stores logs in `logs/train_llamafactory.log`
- emits a `ModelArtifact` pointing at the trained output directory

`eval_model` is now structured as a generic runtime stage with pluggable evaluation backends.
The framework layer handles artifact selection, split selection, generation, and result persistence.
Benchmark-specific prompt building and scoring live in backend modules. The first concrete backend is
`gsm8k`, which writes exact-match results plus per-example predictions.

If you do not pass a custom `llm_client`, DataOS now expects the agent layer to use the
Claude Agent SDK-backed client. Set:

- `STARTEAM_BASE_URL`
- `STARTEAM_API_KEY`
- optional `STARTEAM_MODEL` defaulting to `glm-5`
- optional `DATAOS_CLAUDE_CLI_PATH` pointing to the Claude CLI binary

DataOS still uses the same fixed runtime. The only change is that
plan/review JSON generation goes through Claude Agent SDK via a local Anthropic-to-OpenAI bridge.

## Validation

```bash
uv run pytest
```
