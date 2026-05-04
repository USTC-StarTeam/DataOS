# DataOS Operator Checklist

A practical reference for adding or modifying an operator. Pair with `SKILL.md` for the high-level mental model.

## Read First

- `dataos/core/operator.py` — the `Operator` ABC.
- `dataos/core/artifact.py` — artifact types and `make_artifact_id`.
- `dataos/core/sandbox.py` — output dirs and `log` / `write_json`.
- `dataos/operators/__init__.py` — composition root for built-ins.
- A nearby reference operator: `dataos/operators/dedup.py` (key-based filter) or `dataos/operators/sample.py` (split-aware shaping).

## Operator Contract

```python
class Operator(ABC):
    name: str = ""        # stable identifier seen by the LLM and registry lookups
    description: str = "" # short human-readable description

    @abstractmethod
    def run(
        self,
        inputs: list[Artifact],
        sandbox: Sandbox,
        **kwargs,
    ) -> Artifact | list[Artifact]: ...
```

Rules of the road:

- `name` must be unique within the registry and stable across versions (the LLM plans by name, templates reference by name).
- `run` is deterministic given the same `inputs` and `kwargs`; take a `seed` if you need randomness.
- All outputs go through typed `Artifact` subclasses. No raw filesystem paths leak out.
- Validate `kwargs` and raise `ValueError(f"{self.name}: ...")` early.

## Operator Skeleton

```python
from __future__ import annotations

from datasets import Dataset, DatasetDict

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
    description = "One sentence: what deterministic transform this performs."

    def run(self, inputs: list[Artifact], sandbox: Sandbox, **kwargs) -> DatasetArtifact:
        source = _find_latest_dataset(inputs)
        param = self._validate_params(kwargs)

        dataset_obj = source.load()
        if isinstance(dataset_obj, DatasetDict):
            transformed = DatasetDict({
                split: self._transform(split_dataset, param)
                for split, split_dataset in dataset_obj.items()
            })
        else:
            transformed = self._transform(dataset_obj, param)

        artifact_id = make_artifact_id("dataset")
        target = DatasetArtifact(
            artifact_id=artifact_id,
            name=f"{source.name}-my-operator",
            path=sandbox.artifact_dir("dataset", artifact_id),
            parent_ids=[source.artifact_id],
            producer=self.name,
            meta={"my_operator_param": param},
        )
        target.save(transformed)
        sandbox.log(
            self.name,
            f"source={source.artifact_id}\noutput={target.artifact_id}\nparam={param}\n",
        )
        return target

    @staticmethod
    def _validate_params(kwargs: dict) -> int:
        value = kwargs.get("my_param")
        if value is None:
            raise ValueError("my_operator: 'my_param' is required")
        return int(value)

    @staticmethod
    def _transform(dataset: Dataset, param: int) -> Dataset:
        ...
```

## Important Details

- The runtime passes the **full accumulated artifact list** into every operator (initial dataset plus everything produced earlier in the experiment). For derived data operators, scanning `inputs` in reverse picks the latest upstream of a given type.
- Use `DatasetArtifact.save(dataset_obj)` instead of `save_to_disk(...)`. `save` refreshes `meta.json` (`hf_object_type`, `splits`, `num_rows`, `columns`, `features`) and keeps in-memory state in sync with disk.
- `meta` should be small, explicit, and JSON-serializable. Record:
  - the parameters that determined the transform (after defaults applied),
  - per-split kept/removed counts when shape changes,
  - any seeds or strategy choices.
- Logs go through `sandbox.log("<operator_name>", text)` — they land under `sandbox/logs/<operator_name>.log`. Keep entries terse and grep-friendly: `key=value` lines.
- Prefer one clear output artifact. Return a list only when the operator genuinely produces multiple typed outputs (rare for data operators).

## Registration

The composition root for built-in operators is `dataos/operators/__init__.py`:

```python
def builtin_operators() -> list[Operator]:
    return [
        DedupOperator(),
        FormatConvertOperator(),
        ImportDatasetOperator(),
        SampleOperator(),
        # add MyOperator() here in the order you want the agent to see it
    ]
```

When you add an operator:

1. Implement it under `dataos/operators/<name>.py`.
2. Import it in `dataos/operators/__init__.py`, add it to `__all__`, and append to `builtin_operators()`.
3. Nothing else needs to change. `dataos/runtime/`, `dataos/core/`, and `dataos/cli.py` discover operators only through `build_default_registry()` and `registry.get(name)`.

Notes:

- Order matters. `LLMPlanningAgent._fallback_plan()` picks the first registered operator when the LLM plan is unusable. Put the safest, most no-op-friendly operator first.
- Do **not** register `train_llamafactory` or `eval_model` — those are runtime stages, not operators. Operators do not own training or evaluation logic.

## Templates and the LLM-Facing Surface

- `pipeline_hint` in task templates (`dataos/templates/*.json`) is just a hint; values must be names of currently registered operators. After a rename or removal, sweep templates so hints still resolve.
- The plan-prompt example in `dataos/prompts/agent_rules.py` cites a real operator by name. If you delete the cited one, update the example.

## Test Expectations

Add a focused test for the new or changed operator:

- File name: `tests/test_operator_<name>.py` or `tests/test_operators_<name>.py` (these patterns match the project's test allowlist).
- Cover:
  - happy path on a flat `Dataset`,
  - happy path on a `DatasetDict`,
  - parameter validation (missing required fields, invalid choices),
  - artifact metadata (`parent_ids`, `producer`, `meta` keys),
  - log file existence under `sandbox/logs/`.

If registry composition or default order changed, also touch `tests/test_pipeline_smoke.py` so the smoke pipeline stays valid.

Recommended command order:

```bash
uv run pytest tests/test_operator_<name>.py
uv run pytest --ignore=tests/test_eval_livecodebench.py
```

(`tests/test_eval_livecodebench.py` has a pre-existing collection error unrelated to operator work; ignore it until that file is fixed in `dataos/eval/`.)

## Common Pitfalls

- **Importing a concrete operator class from outside `dataos/operators/`.** This re-introduces the coupling we removed. Always go through the registry.
- **Mutating the input artifact in place.** Always produce a fresh `artifact_id` and a new directory under `sandbox.artifact_dir(...)`.
- **Forgetting `parent_ids`.** Lineage is what makes the experiment report and replay work. Always set it.
- **Putting non-deterministic logic without a seed.** Pipelines must be reproducible.
- **Stuffing training, eval, model loading, or network calls into an operator.** Those belong in runtime stages or dedicated services. An operator is `data -> data`.
- **Dropping rows silently.** Record removals in `meta` and the step log so reviewers can audit.
