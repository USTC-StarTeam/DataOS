---
name: write-dataos-task-template
description: Draft or revise DataOS task template JSON files that match the current TaskTemplate schema, fixed runtime, and registered operators in this repository. Use when asked to write a new task JSON, modify `dataos/templates/*.json`, choose between stub and LLaMA-Factory training configs, or shape `eval_config` for an existing DataOS eval task.
---

# Write DataOS Task Template

Use this skill when the output should be a valid `TaskTemplate` payload, not just a prose description of a task.

## Follow This Order

1. Start from the current schema, not memory.
   - Read `dataos/core/task.py` or [task-template-guide.md](references/task-template-guide.md).
   - Every task template must include all required fields; there are no optional top-level schema fields in `TaskTemplate.from_dict()`.

2. Reuse the closest existing template.
   - Use `dataos/templates/sft_v1.json` for a stub training and stub eval baseline.
   - Use `dataos/templates/sft_llamafactory_v1.json` for LLaMA-Factory training with stub eval.
   - Use `dataos/templates/gsm8k_llamafactory_v1.json` for a concrete native eval example.

3. Keep the template aligned with the fixed runtime.
   - `pipeline_hint` should name only data operators, not runtime stages.
   - The current default registered operators are `clean_data` and `synthesize_data`.
   - Do not put `train_llamafactory` or `eval_model` into `pipeline_hint`.

4. Choose `train_config` deliberately.
   - For fast local validation, use `{"mode": "stub"}`.
   - For real training, use `{"mode": "llamafactory", ...}` with `model_name_or_path`, `template`, and a valid `finetuning_type`.
   - Keep LLaMA-Factory overrides inside `extra_args` unless the runtime explicitly supports a top-level train config key.

5. Choose `eval_config` deliberately.
   - Use `provider: "stub"` for a placeholder eval.
   - Use `provider: "native"` when targeting an in-repo eval task.
   - Keep benchmark-specific settings under `eval_config.task.config`, not as new top-level eval keys.

6. Keep the contract honest.
   - `input_description` and `constraints` should reflect the actual dataset contract: Hugging Face `Dataset` or `DatasetDict` directories saved by `save_to_disk`.
   - `budget.max_experiments` and `max_rounds` should not imply multi-round runtime behavior that the current code does not execute.

## Common Outputs

- For a new generic SFT run, emit a template close to `sft_v1.json` or `sft_llamafactory_v1.json`.
- For a benchmarked training run, emit a template close to `gsm8k_llamafactory_v1.json`.
- For edits, preserve the user's intent but normalize fields back to what `TaskTemplate.from_dict()` accepts today.

## Boundaries

- Do not invent new top-level task template keys unless the repository schema is being changed explicitly.
- Do not claim multi-round execution from `max_rounds`; current runtime still performs one experiment run.
- If the requested eval target is not already implemented under `dataos/eval/tasks/`, hand off conceptually to `$add-dataos-eval-task` rather than fabricating a config for a task that does not exist.
