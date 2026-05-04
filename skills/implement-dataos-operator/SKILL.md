---
name: implement-dataos-operator
description: Add or modify DataOS operators. Use when implementing a new operator, changing operator behavior, registering an operator in the default pipeline, or explaining operator design. Operators are deterministic Hugging-Face-dataset-in / artifact-out transforms; training and evaluation are runtime stages, not operators.
---

# Implement DataOS Operator

Use this skill when work touches `dataos/core/operator.py`, `dataos/operators/`, or default registry composition in `dataos/operators/__init__.py`.

## Mental Model

- An **operator** is a deterministic, sandboxed function `Artifact[] -> Artifact (or Artifact[])`. It transforms data and produces typed artifacts. Nothing else.
- The **runtime** (`dataos/runtime/`) and **core** (`dataos/core/`) layers know nothing about specific operators. They depend only on the `Operator` ABC and the `OperatorRegistry`.
- The **composition root** for built-in operators is `dataos/operators/__init__.py`. To add or remove a built-in operator, only this file needs to change — runtime, CLI, and agent code are operator-agnostic.
- **Training and evaluation are not operators.** They run as fixed runtime stages (`dataos/train/`, `dataos/eval/`) after the planned operator pipeline. Do not wrap them as operators.

## Follow This Order

1. Read the contract first.
   - `dataos/core/operator.py` — the `Operator` ABC.
   - `dataos/core/artifact.py` — `DatasetArtifact`, `ModelArtifact`, `EvalArtifact`, `ResultArtifact`, `ExperimentArtifact`, and `make_artifact_id`.
   - `dataos/core/sandbox.py` — output directories and logging.
   - One nearby operator (`dataos/operators/dedup.py` or `sample.py`) for shape and conventions.
   - Then [operator-checklist.md](references/operator-checklist.md).

2. Preserve the operator contract.
   - Subclass `Operator`, set `name` and `description` as class attributes.
   - Implement `run(self, inputs, sandbox, **kwargs)`.
   - Return one typed `Artifact` (or a list of typed artifacts). No raw paths, no side effects outside the sandbox.
   - Be deterministic given the same inputs and `kwargs`. Take a `seed` parameter when randomness is required.

3. Choose inputs deliberately.
   - The runtime hands every accumulated artifact to every operator. For a derived data operator, scan `inputs` in reverse to pick the latest upstream artifact of the right type.
   - Validate required `kwargs` early and raise `ValueError` with a clear message keyed by the operator name (`f"{self.name} requires ..."`).
   - Set `parent_ids` on the output artifact to the upstream `artifact_id`s actually consumed.

4. Write inside the sandbox.
   - Allocate output directories with `sandbox.artifact_dir("dataset", artifact_id)` (or other type prefix).
   - Persist datasets via `DatasetArtifact.save(dataset_obj)`. Do not call `save_to_disk` directly — `save` keeps `meta.json` and on-disk state aligned.
   - Append a step log via `sandbox.log("<operator_name>", text)` summarizing inputs, outputs, and key counts.
   - Keep `meta` small, explicit, JSON-serializable. Record the parameters that determined the transform plus shape stats (kept/removed counts).

5. Register the operator.
   - Add it to the `builtin_operators()` list in `dataos/operators/__init__.py` and re-export the class in `__all__`.
   - **Do not edit** `dataos/runtime/runner.py`, `dataos/core/agent.py`, or any other file just to wire a new operator. If you find yourself doing so, the decoupling is being violated.
   - Order in `builtin_operators()` is the order the agent sees and is also what `_fallback_plan()` picks first when the LLM plan is invalid. Put the most generally safe operator first.

6. Test at the right level.
   - Add `tests/test_operator_<name>.py` (or `tests/test_operators_<name>.py`) covering happy path, parameter validation, and metadata.
   - If registry composition or default order changed, also touch a runtime smoke test (`tests/test_pipeline_smoke.py`).
   - Run the focused file first, then the full suite:
     ```bash
     uv run pytest tests/test_operator_<name>.py
     uv run pytest --ignore=tests/test_eval_livecodebench.py
     ```

## Boundaries

- Do not import concrete operator classes from `dataos/runtime/`, `dataos/core/`, `dataos/cli.py`, `dataos/agent/`, or `dataos/research/`. Go through `dataos.operators.build_default_registry()` and `registry.get(name)` instead.
- Do not introduce new dataset storage formats. All datasets are Hugging Face `Dataset` / `DatasetDict` `save_to_disk` directories wrapped in a `DatasetArtifact`.
- Do not add new artifact types without updating `dataos/core/artifact.py` and the local store together.
- Do not register training, evaluation, or any side-effecting stage as an operator.
- Do not silently drop rows or splits. Always record removed/kept counts in `meta` and `sandbox.log`.
