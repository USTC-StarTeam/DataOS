# DataOS Main Document

Last updated: 2026-05-10.

This is the main maintained architecture overview in `docs/`.

## Current Scope

DataOS is now scoped as a local-first single-experiment runtime for SFT data
workflows. The stable architecture is intentionally narrow:

- Hugging Face `Dataset` / `DatasetDict` directories are the internal dataset
  artifact format.
- `run_task()` executes one experiment inside a local experiment directory.
- The runtime planning agent plans operator steps and reviews the finished
  experiment.
- Operators execute deterministically in local Python and emit typed artifacts.
- A small subset of evaluator / generator operators may optionally call an
  OpenAI-compatible endpoint when the task explicitly needs LLM scoring or data
  augmentation.
- Training and evaluation are fixed runtime stages.
- Evaluation keeps lm-eval backend compatibility and a small natural-language
  eval helper.

There is no research layer, daemon loop, project queue, resource scheduler, or
asynchronous agent control plane.

## Agent Boundary

DataOS keeps one agent-like component:

| Component | Location | Scope | Owns |
|---|---|---|---|
| Main agent loop | `src/dataos/agent/` | A budgeted experiment search | state tree, node selection, plan/review calls |
| Runtime | `src/dataos/runtime/` | One experiment execution | local/sub-agent executors and report collection |

The agent does not own projects, queues, resources, leases, background workers,
or persistent control-plane state.

## Runtime Flow

The autonomous experiment entrypoint is `dataos.agent.loop.run_auto()`.

1. Materialize one or more dataset inputs into experiment-managed Hugging Face dataset directories.
2. Ask `LLMPlanningAgent` for operator steps.
3. Execute registered data operators deterministically.
4. Run fixed runtime stages:
   - `train_llamafactory`
   - `eval_model`
5. Finalize the experiment artifact.
6. Ask the same runtime agent for a structured review.
7. Persist the final report and artifacts inside the same experiment directory.

The LLM plans and reviews only. Execution stays in local code.

## Operator Registry

The operator catalog is provided by the external `dataos_operator` package and
surfaced through `dataos.core.catalog.catalog()`. Common operator categories are:

- mapper
  - `map_to_alpaca`
  - `map_to_sharegpt`
- selector
  - `select_random_subset`
  - `select_top_by_score`
- filter
  - `filter_exact_dedup`
  - `filter_by_text_length`
- evaluator
  - `evaluate_text_quality`
  - `evaluate_llm_pair`
- generator
  - `generate_rewritten_query`
  - `generate_rewritten_answer`

These are data-stage operators only. Training and evaluation stay outside the
operator catalog as fixed runtime/integration stages.

## Module Map

| Path | Role |
|---|---|
| `src/dataos/agent/` | Budgeted main-agent loop, state tree, node selection, planner/reviewer prompts and schemas. |
| `src/dataos/core/` | Artifact model, project/study/experiment objects, workspace layout, reports, ids, IO, provenance. |
| `src/dataos/runtime/` | Experiment execution helpers: local fan-out, sub-agent executor, data preparation, frozen eval selection. |
| `src/dataos/integrations/` | Concrete external integrations: Claude Code, PostTrainBench, LLaMA-Factory, Slurm, lm-eval. |
| `configs/` | Run configs, including `configs/runs/ptb-h100-qwen-gsm8k.json`. |
| `resources/data/` | HF datasets and benchmark data resources. |
| `resources/code/` | Benchmark and harness code resources used by runtime/sub-agents. |
| `scripts/` | Thin command-line helpers such as `scripts/run_one.py` and `scripts/doctor.py`. |

## Eval Scope

`src/dataos/integrations/lm_eval/` keeps the lm-eval integration:

- `dataos.integrations.lm_eval.runner.run_eval_stage()` persists `EvalArtifact` objects from
  the runtime's fixed `eval_model` stage.

## Current CLI Shape

The CLI is intentionally small:

```bash
uv run dataos configs/runs/ptb-h100-qwen-gsm8k.json
```

The run config is the only user-facing interface for PTB/H100 runs. It records
the run id, workspace, goal, bench, base model path, time budget, requested node
/ GPU / CPU resources, and fixed remote paths. Runtime defaults are expanded
into `resolved_config.json` inside the run workspace.

## Stable Paper Framing

The stable framing for the paper is:

1. DataOS standardizes local SFT experiment artifacts around Hugging Face
   datasets, typed artifacts, and single-root experiment directories.
2. Planning and review are still the main LLM boundary, while a small subset of
   evaluator / generator operators can optionally call OpenAI-compatible
   endpoints when the task explicitly requires it.
3. Local code still owns operator orchestration, training, evaluation, and
   artifact persistence.
4. lm-eval compatibility gives evaluation coverage without turning evaluation
   planning into another agent runtime.
