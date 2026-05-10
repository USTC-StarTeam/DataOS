# DataOS Main Document

Last updated: 2026-05-10.

This is the only maintained architecture document in `docs/`.

## Current Scope

DataOS is now scoped as a local-first single-experiment runtime for SFT data
workflows. The stable architecture is intentionally narrow:

- Hugging Face `Dataset` / `DatasetDict` directories are the internal dataset
  artifact format.
- `run_task()` executes one experiment inside a local sandbox.
- The runtime planning agent plans operator steps and reviews the finished
  experiment.
- Operators execute deterministically in local Python and emit typed artifacts.
- Training and evaluation are fixed runtime stages.
- Evaluation keeps lm-eval backend compatibility and a small natural-language
  eval helper.

There is no research layer, daemon loop, project queue, resource scheduler, or
asynchronous agent control plane.

## Agent Boundary

DataOS keeps one agent-like component:

| Component | Location | Scope | Owns |
|---|---|---|---|
| Runtime planning agent | `dataos/core/agent.py` | One experiment | `plan()`, `review()`, `plan.json`, `review.json`, operator-step selection |

The agent does not own projects, queues, resources, leases, background workers,
or persistent control-plane state.

## Runtime Flow

The authoritative experiment entrypoint is `dataos.runtime.run_task()`.

1. Materialize the input Hugging Face dataset directory into the sandbox.
2. Ask `LLMPlanningAgent` for operator steps.
3. Execute registered data operators deterministically.
4. Run fixed runtime stages:
   - `train_llamafactory`
   - `eval_model`
5. Finalize the experiment artifact.
6. Ask the same runtime agent for a structured review.
7. Archive the experiment through `LocalArtifactStore`.

The LLM plans and reviews only. Execution stays in local code.

## Operator Registry

The default registry is composed from `dataos/operators/__init__.py`. It contains
deterministic data operators such as:

- `dedup`
- `sample`
- `schema_guard`
- `pii_redact`
- `length_filter`
- `lang_filter`
- `quality_filter`
- `category_tag`
- `category_balance`
- `curriculum_sort`
- `mixture`
- `instruction_synthesis_template`
- `format_convert`
- `import_dataset`
- `less`

These are data-stage operators only. `train_llamafactory` and `eval_model` stay
outside the registry as fixed runtime stages.

## Module Map

| Path | Role |
|---|---|
| `dataos/core/` | Artifact model, sandbox, experiment state, task schema, operator interface, planning agent. |
| `dataos/runtime/` | `run_task()` orchestration and fixed runtime stage wiring. |
| `dataos/operators/` | Deterministic dataset operators. |
| `dataos/train/` | LLaMA-Factory training runtime stage. |
| `dataos/eval/` | lm-eval bridge, adapters, eval config, NL-to-lm-eval helper. |
| `dataos/llm/` | LLM client interfaces and OpenAI-compatible implementation. |
| `dataos/prompts/` | Planning and review prompt builders. |
| `dataos/templates/` | Example task templates and import specs. |
| `dataos/cli.py` | Minimal CLI; currently keeps `data import`. |

## Eval Scope

`dataos/eval/` keeps two responsibilities:

- `dataos.eval.runner.run_eval_stage()` persists `EvalArtifact` objects from
  the runtime's fixed `eval_model` stage.
- `dataos.eval.planner.plan_eval()` maps a natural-language eval goal to
  lm-eval task names using keyword intent by default and optional LLM intent
  when explicitly requested.

The eval planner is a helper for selecting lm-eval tasks. It is not a project
agent.

## Current CLI Shape

The CLI is intentionally small:

```bash
dataos data import --workspace workspace --spec import.json
```

End-to-end experiment execution should use the Python API:

```python
from dataos.runtime import run_task

result = run_task(
    task_template="dataos/templates/sft_v1.json",
    dataset_path="/path/to/hf_dataset",
    workspace_dir="workspace",
)
```

## Stable Paper Framing

The stable framing for the paper is:

1. DataOS standardizes local SFT experiment artifacts around Hugging Face
   datasets, typed artifacts, and sandboxed experiment directories.
2. LLM calls are confined to planning and reviewing.
3. Local deterministic code owns operator execution, training, evaluation, and
   artifact persistence.
4. lm-eval compatibility gives evaluation coverage without turning evaluation
   planning into another agent runtime.
