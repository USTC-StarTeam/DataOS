# DataOS Runtime Reference

## Setup

Install the repo with development dependencies:

```bash
uv sync --extra dev
```

Install the training stack only when the request actually needs real LLaMA-Factory runs:

```bash
uv sync --extra dev --extra train
```

## Main Entrypoint

Use `dataos.runtime.run_task(...)` for end-to-end local runs.

Minimal example:

```python
from pathlib import Path

from dataos.runtime import run_task

result = run_task(
    task_template=Path("dataos/templates/sft_v1.json"),
    dataset_path=Path("/path/to/hf_dataset"),
    workspace_dir=Path("workspace"),
)
```

If the caller does not pass `llm_client`, the current code in `dataos/runtime/runner.py` expects:

- `STARTEAM_API_KEY`
- `STARTEAM_BASE_URL`
- optional `STARTEAM_MODEL`

## Dataset Contract

- Input datasets must be Hugging Face `Dataset` or `DatasetDict` directories created by `save_to_disk`.
- `DatasetArtifact` only supports `meta["format"] == "hf_datasets"`.
- `run_task()` copies the source dataset into `sandbox/inputs/<dataset-id>/` and materializes the initial `DatasetArtifact` there.

Training currently expects dataset rows with:

- `instruction`
- `response`
- optional `input`
- optional `system`
- optional `history`

Evaluation dataset fields depend on the chosen eval task.

## TaskTemplate Contract

`dataos/core/task.py` requires these fields:

- `task_id`
- `name`
- `goal`
- `input_description`
- `output_description`
- `constraints`
- `budget`
- `max_rounds`
- `pipeline_hint`
- `train_config`
- `eval_config`

Useful examples live under `dataos/templates/`:

- `sft_v1.json`
- `sft_llamafactory_v1.json`
- `gsm8k_llamafactory_v1.json`

## Runtime Sequence

`run_task()` performs work in this order:

1. Prepare the local artifact store under `workspace/artifacts/`
2. Load the task template
3. Prepare a sandbox under `workspace/sandboxes/<exp-id>/`
4. Materialize the initial dataset artifact
5. Build the default operator registry
6. Ask the planning agent for operator steps
7. Execute planned operators
8. Execute fixed runtime stages:
   - `train_llamafactory`
   - `eval_model`
9. Finalize the experiment and write `report.json`
10. Generate `review.json`
11. Archive the experiment artifact into `workspace/artifacts/experiments/`

## Where To Inspect Output

Within the sandbox root:

- `plan.json`
- `report.json`
- `review.json`
- `inputs/`
- `artifacts/`
- `logs/`
- `tmp/`
- `code/`

Within the workspace root:

- `artifacts/datasets/`
- `artifacts/models/`
- `artifacts/evals/`
- `artifacts/results/`
- `artifacts/experiments/`

## Fixed Runtime Facts

- Default operator registry order is `clean_data`, then `synthesize_data`.
- The planning agent fallback plan assumes that same order.
- `train_llamafactory` and `eval_model` are runtime stages, not registered operators.
