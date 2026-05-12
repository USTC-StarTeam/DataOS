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
| Runtime planning agent | `dataos/core/agent.py` | One experiment | `plan()`, `review()`, `plan.json`, `review.json`, operator-step selection |

The agent does not own projects, queues, resources, leases, background workers,
or persistent control-plane state.

## Runtime Flow

The authoritative experiment entrypoint is `dataos.runtime.run_task()`.

1. Materialize one or more dataset inputs into sandbox-managed Hugging Face dataset directories.
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

The default registry is composed from `dataos/operators/__init__.py`. It now
groups operators by category:

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

These are data-stage operators only. `train_llamafactory` and `eval_model` stay
outside the registry as fixed runtime stages.

## Module Map

| Path | Role |
|---|---|
| `dataos/core/` | Artifact model, sandbox, experiment state, task schema, operator interface, planning agent. |
| `dataos/runtime/` | `run_task()` orchestration and fixed runtime stage wiring. |
| `dataos/operators/` | Category-organized dataset operators plus shared operator helpers. |
| `dataos/train/` | LLaMA-Factory training runtime stage. |
| `dataos/eval/` | lm-eval bridge, adapters, eval config, NL-to-lm-eval helper. |
| `dataos/llm/` | LLM client interfaces and OpenAI-compatible implementation. |
| `dataos/prompts/` | Planning and review prompt builders. |
| `dataos/templates/` | Example task templates and import specs. |
| `dataos/cli.py` | CLI entrypoints for operator inspection and `dataos train`. |

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
uv run dataos operators list

uv run python - <<'PY'
from pathlib import Path
import shutil

from datasets import load_dataset
from huggingface_hub import hf_hub_download

root = Path("datasets")
root.mkdir(parents=True, exist_ok=True)

gsm8k = load_dataset("openai/gsm8k", "main")
gsm8k_dir = root / "openai__gsm8k_main"
if gsm8k_dir.exists():
    shutil.rmtree(gsm8k_dir)
gsm8k.save_to_disk(str(gsm8k_dir))

alpaca_json = hf_hub_download(
    repo_id="tatsu-lab/alpaca_eval",
    repo_type="dataset",
    filename="alpaca_eval.json",
)
alpaca = load_dataset("json", data_files=alpaca_json)
alpaca_dir = root / "tatsu_lab__alpaca_eval"
if alpaca_dir.exists():
    shutil.rmtree(alpaca_dir)
alpaca.save_to_disk(str(alpaca_dir))
PY

uv run dataos train \
  --task dataos/templates/alpaca_gsm8k_stub_v1.json \
  --dataset datasets/tatsu_lab__alpaca_eval,datasets/openai__gsm8k_main \
  --workspace workspace
```

If `uv` is installed under `~/.local/bin`, add it to `PATH` before running the
CLI:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

End-to-end experiment execution should use the Python API:

```python
from dataos.runtime import run_task

result = run_task(
    task_template="dataos/templates/alpaca_gsm8k_stub_v1.json",
    datasets=[
        "datasets/tatsu_lab__alpaca_eval",
        "datasets/openai__gsm8k_main",
    ],
    workspace_dir="workspace",
)
```

## Stable Paper Framing

The stable framing for the paper is:

1. DataOS standardizes local SFT experiment artifacts around Hugging Face
   datasets, typed artifacts, and sandboxed experiment directories.
2. Planning and review are still the main LLM boundary, while a small subset of
   evaluator / generator operators can optionally call OpenAI-compatible
   endpoints when the task explicitly requires it.
3. Local code still owns operator orchestration, training, evaluation, and
   artifact persistence.
4. lm-eval compatibility gives evaluation coverage without turning evaluation
   planning into another agent runtime.
