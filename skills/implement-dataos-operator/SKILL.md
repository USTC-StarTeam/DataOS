---
name: implement-dataos-operator
description: Add or modify DataOS operators under the fixed runtime while preserving Hugging Face dataset artifacts, sandbox output conventions, and registry and fallback coherence. Use when asked to implement a new operator, change operator behavior, wire an operator into the default pipeline, or explain operator design in this repository.
---

# Implement DataOS Operator

Use this skill when the work touches `dataos/core/operator.py`, `dataos/operators/`, or default registry wiring.

## Follow This Order

1. Read the contract first.
   - Inspect `dataos/core/operator.py`, `dataos/runtime/runner.py`, and one nearby operator implementation.
   - Then read [operator-checklist.md](references/operator-checklist.md).

2. Preserve the runtime model.
   - Implement the operator as deterministic local code.
   - Accept `inputs`, `sandbox`, and `**kwargs`.
   - Emit typed artifacts only.
   - Do not move training or evaluation logic into operators. `train_llamafactory` and `eval_model` stay runtime stages.

3. Choose inputs deliberately.
   - Prefer the latest relevant upstream artifact from `inputs`.
   - Keep `parent_ids` explicit.
   - If the operator produces a dataset, persist it with `DatasetArtifact.save(...)` so metadata stays in sync with disk.

4. Write inside the sandbox.
   - Use `sandbox.artifact_dir(...)` for output directories.
   - Use `sandbox.log(...)` for stable step logs.
   - Keep metadata JSON-serializable.

5. Wire the runtime when needed.
   - Register standard pipeline operators in `build_default_registry()` in `dataos/runtime/runner.py`.
   - If the default operator set or order changes, update `LLMPlanningAgent._fallback_plan()` in `dataos/core/agent.py` to match.

6. Test at the right level.
   - Add a focused operator test for transformation behavior and metadata.
   - Add or update a runtime-level test if registry wiring or default pipeline behavior changed.
   - Run targeted pytest first, then the full suite.

## Boundaries

- Do not introduce new dataset storage formats.
- Do not add new artifact types without updating the central artifact model and local store together.
- Do not register `train_llamafactory` or `eval_model` as operators.
