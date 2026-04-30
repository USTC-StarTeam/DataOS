# DataOS Task Template Guide

## Required Top-Level Fields

`dataos/core/task.py` currently requires every task JSON to include:

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

There are no optional top-level fields in `TaskTemplate.from_dict()`. If one is missing, loading fails.

## Safe Starting Points

Use the closest template under `dataos/templates/`:

- `sft_v1.json`
  - stub training
  - stub eval
  - generic SFT baseline

- `sft_llamafactory_v1.json`
  - real LLaMA-Factory training
  - stub eval
  - generic SFT training baseline

- `gsm8k_llamafactory_v1.json`
  - LLaMA-Factory training
  - native eval example
  - concrete benchmark-style template

## Current Runtime Constraints

- Input datasets must be Hugging Face `Dataset` or `DatasetDict` directories created by `save_to_disk`.
- `pipeline_hint` should contain only data operators.
- The current default operator set is:
  - `clean_data`
  - `synthesize_data`
- `train_llamafactory` and `eval_model` are fixed runtime stages and should not appear in `pipeline_hint`.

## train_config Patterns

### Fast local validation

```json
{
  "mode": "stub"
}
```

### Real LLaMA-Factory training

```json
{
  "mode": "llamafactory",
  "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
  "template": "qwen",
  "finetuning_type": "lora",
  "extra_args": {
    "cutoff_len": 512,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 0.0001,
    "num_train_epochs": 1.0,
    "logging_steps": 1,
    "save_steps": 20
  }
}
```

Notes:

- Supported runtime modes are currently `stub` and `llamafactory`.
- For `llamafactory`, the runtime requires:
  - `model_name_or_path`
  - `template`
  - valid `finetuning_type`
- Step overrides are constrained by `dataos/train/llamafactory.py`; do not add arbitrary top-level keys.

## eval_config Patterns

### Stub eval

```json
{
  "provider": "stub",
  "provider_config": {
    "score": 0.5
  }
}
```

### Native eval

```json
{
  "provider": "native",
  "task": {
    "name": "gsm8k",
    "config": {}
  },
  "data": {
    "split": "test"
  },
  "adapter": {
    "name": "transformers"
  },
  "request": {
    "type": "generate_until",
    "batch_size": 8,
    "max_new_tokens": 256
  }
}
```

Allowed top-level eval keys are currently:

- `provider`
- `provider_config`
- `task`
- `data`
- `adapter`
- `request`
- `artifacts`

Put benchmark-specific field mapping and knobs inside `eval_config.task.config`.

## Authoring Heuristics

- Keep `constraints.dataset_format` aligned with `hf_datasets`.
- Keep `constraints.execution_mode` aligned with `fixed_runtime`.
- Use `budget.max_experiments` as planning metadata, not as a promise that the runtime loops today.
- Keep `max_rounds` honest; current runtime executes one experiment run.
- Keep `output_description` focused on actual artifacts and reports the runtime can produce.
