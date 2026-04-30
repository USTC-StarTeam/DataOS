---
name: add-dataos-eval-task
description: Extend DataOS evaluation with a new native benchmark task, dataset contract, scoring logic, or dataset variant while keeping eval_model as a fixed runtime stage. Use when asked to add an eval dataset, benchmark, or scoring task under dataos/eval, or to explain how eval_config should be shaped for a new task in this repository.
---

# Add DataOS Eval Task

Use this skill when the request is really "extend the current eval system" rather than "add another operator."

## Follow This Order

1. Confirm the extension point.
   - New benchmarks normally belong under `dataos/eval/tasks/`, not as new operators.
   - `eval_model` stays a fixed runtime stage driven by `task.eval_config`.
   - Read [eval-extension.md](references/eval-extension.md).

2. Decide whether the work needs a new task class at all.
   - If an existing task can handle the new dataset by adding field-name overrides in `eval_config.task.config`, prefer that path.
   - If the request needs new prompt construction, normalization, or scoring semantics, add a new task class.

3. Implement the task contract.
   - Add a task class with `name`, `version`, `supported_request_types`, `default_split()`, `build_requests()`, and `score_responses()`.
   - Keep benchmark-specific field mapping and normalization inside the task or small helpers under `metrics/` or `postprocessors/`.
   - Use `task.config` overrides for dataset-specific field names instead of adding new top-level eval config keys.

4. Register without changing the runtime shape.
   - Register the task in `dataos/eval/registry.py`.
   - Reuse an existing provider and adapter unless the request truly needs a new one.
   - If the task needs sandbox runtime context, implement `set_runtime_context()` and let the native provider inject it.

5. Keep dataset and artifact contracts stable.
   - Evaluation still reads from a Hugging Face `Dataset` or `DatasetDict` artifact saved to disk.
   - Keep result persistence flowing through `run_eval_stage()` and `EvalArtifactStore`; do not invent a parallel persistence path.
   - Do not add a new artifact type for a benchmark.

6. Test both task logic and runtime integration.
   - Add a task-focused test that builds a small `Dataset` and calls the task directly.
   - Add an integration-style test that runs `run_eval_stage(...)` with a `TaskTemplate`.
   - Check score, summary, trace artifacts, and failure messages for missing fields or splits.

## Boundaries

- Prefer `provider="native"` for in-repo benchmarks.
- Keep error messages concrete about required dataset fields.
- Prefer small reusable postprocessors or metrics over task logic embedded in adapters or providers.
