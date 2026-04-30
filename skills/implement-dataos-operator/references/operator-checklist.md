# DataOS Operator Checklist

## Read First

- `dataos/core/operator.py`
- `dataos/runtime/runner.py`
- `dataos/core/agent.py`
- a nearby operator in `dataos/operators/`

## Operator Skeleton

Use the local interface from `dataos/core/operator.py`:

```python
from dataos.core.artifact import Artifact, DatasetArtifact, make_artifact_id
from dataos.core.operator import Operator
from dataos.core.sandbox import Sandbox


def _find_latest_dataset(inputs: list[Artifact]) -> DatasetArtifact:
    for artifact in reversed(inputs):
        if isinstance(artifact, DatasetArtifact):
            return artifact
    raise ValueError("my_operator requires at least one DatasetArtifact input")


class MyOperator(Operator):
    name = "my_operator"
    description = "Describe the deterministic transformation."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = _find_latest_dataset(inputs)
        dataset_obj = source.load()
        transformed = ...

        artifact_id = make_artifact_id("dataset")
        target = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-my-operator",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={...},
        )
        target.save(transformed)
        sandbox.log("my_operator", f"source={source.artifact_id}\noutput={target.artifact_id}\n")
        return target
```

## Important Details

- Runtime passes the full accumulated artifact list into each operator. For derived data operators, scanning `inputs` in reverse is usually the correct starting point.
- Use `DatasetArtifact.save(...)` instead of calling `save_to_disk(...)` directly so metadata is refreshed and `meta.json` stays aligned.
- Keep `meta` small, explicit, and JSON-serializable.
- Write logs through `sandbox.log(...)`; logs belong under `sandbox/logs/`.
- Prefer one clear output artifact unless the operator genuinely produces multiple typed outputs.

## Registry And Fallback

When a new operator should participate in the default pipeline:

1. Register it in `build_default_registry()` in `dataos/runtime/runner.py`.
2. Update `LLMPlanningAgent._fallback_plan()` in `dataos/core/agent.py` if the default pipeline order changed.
3. Keep the registered order and fallback order consistent.

Do not register:

- `train_llamafactory`
- `eval_model`

Those remain fixed runtime stages.

## Test Expectations

For a new or changed operator, add:

- a focused test for the transformation logic
- assertions on artifact metadata and parent lineage
- a runtime-level test if default registry wiring changed

Recommended command order:

```bash
uv run pytest tests/<closest_test_file>.py
uv run pytest
```
