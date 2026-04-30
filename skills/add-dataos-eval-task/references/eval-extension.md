# DataOS Eval Extension Reference

## Current Architecture

Evaluation runs through the fixed runtime stage `eval_model`:

1. `run_task()` calls `run_eval_stage(...)`
2. `run_eval_stage(...)` resolves `task.eval_config`
3. The provider is loaded from `dataos/eval/registry.py`
4. Native provider loads a task from `dataos/eval/tasks/`
5. The provider persists trace artifacts and the final `EvalArtifact`

This means new benchmarks usually extend:

- `dataos/eval/tasks/`
- sometimes `dataos/eval/metrics/`
- sometimes `dataos/eval/postprocessors/`
- `dataos/eval/registry.py`
- `tests/test_eval_<task>.py`

## Allowed eval_config Shape

`dataos/eval/config.py` currently allows only these top-level keys:

- `provider`
- `provider_config`
- `task`
- `data`
- `adapter`
- `request`
- `artifacts`

Put benchmark-specific knobs inside `eval_config.task.config`.

Minimal native config:

```python
{
    "provider": "native",
    "task": {"name": "gsm8k", "config": {}},
    "data": {"split": "test"},
    "adapter": {"name": "transformers", "config": {}},
    "request": {"type": "generate_until"},
}
```

## New Task Checklist

Implement a task class with:

- `name`
- `version`
- `supported_request_types`
- `default_split()`
- `build_requests(dataset, config)`
- `score_responses(dataset, requests, responses, config)`

Use the task to own:

- dataset field requirements
- prompt construction
- reference extraction
- normalization
- scoring semantics

If the task needs sandbox runtime context, optionally add:

- `set_runtime_context(context)`

`NativeEvalProvider` will call it before building requests.

## Dataset Guidance

Stay within the existing artifact contract:

- evaluation input is still a Hugging Face `Dataset` or `DatasetDict`
- data should be saved with `save_to_disk`
- split selection is handled by `eval_config.data.split`

When adding a new dataset variant for an existing task, prefer:

- field-name overrides in `eval_config.task.config`
- slightly broader extraction logic inside the existing task

Only add a new task when the benchmark truly needs different request building or scoring behavior.

## Registration

Add a lazy task factory in `dataos/eval/registry.py` and register it with:

```python
register_task("my_task", _my_task)
```

Reuse existing providers and adapters unless there is a concrete reason not to.
Current in-repo benchmark work should usually stay on:

- provider: `native`
- adapter: `transformers` or `openai_compatible`

## Testing Pattern

For a new task, add both:

1. A direct task test
   - build a tiny `Dataset`
   - call `build_requests(...)`
   - call `score_responses(...)`
   - assert normalized predictions, summary, and score

2. A `run_eval_stage(...)` test
   - build `DatasetArtifact` and `ModelArtifact`
   - build a minimal `TaskTemplate`
   - monkeypatch the adapter response path if needed
   - assert:
     - `EvalArtifact.meta["task"]`
     - `score`
     - `dataset_split`
     - persisted trace artifacts when enabled

Recommended command order:

```bash
uv run pytest tests/test_eval_<task>.py
uv run pytest
```
