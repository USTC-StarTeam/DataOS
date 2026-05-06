# Real Validation Guide

This document describes the two manual validation paths that exercise DataOS beyond unit tests:

- `scripts/run_all_operator_real_checks.py`
- `scripts/run_live_llm_runtime_demo.py`

No API secrets should be written into the repository. Pass credentials only through environment variables in your shell or a local untracked `.env` file.

## Environment Variables

For live OpenAI-compatible calls, set these variables in your shell before running the scripts:

- `DASHSCOPE_BASE_URL` or `STARTEAM_BASE_URL`
- `DASHSCOPE_API_KEY` or `STARTEAM_API_KEY`
- `DASHSCOPE_MODEL` or `STARTEAM_MODEL`

`run_all_operator_real_checks.py` uses them only when `quality_filter` runs in `llm_judge` mode.
`run_live_llm_runtime_demo.py` uses them for planner / review generation through `run_task()`.

## Real Operator Suite

`scripts/run_all_operator_real_checks.py` uses a real `alpaca_eval` JSON export as the source corpus, derives a small canonical SFT dataset from it, and then runs the built-in operators in sequence:

- `import_dataset` from `hf_local`
- `import_dataset` from `parquet`
- `schema_guard`
- `pii_redact`
- `dedup`
- `length_filter`
- `lang_filter`
- `quality_filter`
- `category_tag`
- `category_balance`
- `curriculum_sort`
- `instruction_synthesis_template`
- `sample`
- `format_convert`
- `mixture`
- `less`

The operator flow is real; only the runtime tail is stubbed. After the operator chain finishes, the script emits a stub `ModelArtifact` and a stub `EvalArtifact` so the end-to-end integration shape is still covered without paying the cost of a real training / eval job.

The default `less` mode is `stub`, because the repository does not vendor a LESS checkout. If you want the actual LESS pipeline, pass `--less-mode less --less-repo-path /path/to/LESS`.

```bash
REPO_ROOT=/path/to/DataOS
cd "$REPO_ROOT"

uv sync --extra dev --extra train --extra eval
source .venv/bin/activate

BASE_MODEL=/absolute/path/to/your/local/Qwen2.5-0.5B-snapshot
ALPACA_EVAL_JSON=/absolute/path/to/alpaca_eval.json

python scripts/run_all_operator_real_checks.py \
  --workspace /tmp/dataos-operator-real-checks \
  --alpaca-eval-json "$ALPACA_EVAL_JSON" \
  --base-model "$BASE_MODEL" \
  --quality-filter-mode llm_judge \
  --less-mode stub \
  --overwrite
```

The summary is written to:

```bash
/tmp/dataos-operator-real-checks/operator_real_checks_summary.json
```

## Live LLM Runtime Demo

`scripts/run_live_llm_runtime_demo.py` is intentionally narrower. It does not test the whole operator library. Instead, it validates the "LLM recommends operators" path inside `dataos.runtime.run_task()` against a real `gsm8k` subset.

This script:

- loads real GSM8K rows from JSON
- builds a tiny `DatasetDict` with one intentional duplicate
- constrains the runtime registry to `dedup` and `sample`
- uses a live OpenAI-compatible planner / reviewer
- keeps training and evaluation in stub mode

The point of this demo is to inspect the live plan payload, the executed operator steps, and the archived experiment report without paying for real fine-tuning or real benchmark evaluation.

```bash
REPO_ROOT=/path/to/DataOS
cd "$REPO_ROOT"

uv sync --extra dev --extra train --extra eval
source .venv/bin/activate

GSM8K_JSON=/absolute/path/to/gsm8k_rows.json

python scripts/run_live_llm_runtime_demo.py \
  --workspace /tmp/dataos-live-llm-demo \
  --gsm8k-json "$GSM8K_JSON" \
  --overwrite
```

The summary is written to:

```bash
/tmp/dataos-live-llm-demo/live_llm_runtime_demo_summary.json
```
