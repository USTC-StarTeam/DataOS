---
name: use-dataos
description: Run DataOS end to end on a local Hugging Face Dataset or DatasetDict saved with save_to_disk, prepare TaskTemplate payloads, and inspect sandbox and archived artifacts. Use when asked how to use DataOS, how to run a task locally, how to prepare datasets or task JSON, how to debug plan/report/review files, or how the fixed runtime behaves in this repository.
---

# Use DataOS

Use this skill to answer operational "how do I use DataOS?" questions against the current repository, not against generic platform assumptions.

## Follow This Order

1. Confirm the dataset contract before anything else.
   - Accept only a Hugging Face `Dataset` or `DatasetDict` directory produced by `save_to_disk`.
   - Reject alternate dataset formats unless the user explicitly wants to change the runtime contract.
   - Read [runtime.md](references/runtime.md) for the exact task and artifact expectations.

2. Choose the right entrypoint for the question.
   - Use `dataos.runtime.run_task` for end-to-end execution.
   - Use `dataos/core/task.py` for task schema questions.
   - Use `dataos/runtime/runner.py` for operator order and runtime stages.
   - Use `dataos/core/agent.py` and `dataos/core/experiment.py` for `plan.json`, `report.json`, and `review.json`.

3. Prefer current code over older prose when they differ.
   - The authoritative runtime env lookup is `_build_client_from_env()` in `dataos/runtime/runner.py`.
   - If the caller does not pass `llm_client`, current code expects `STARTEAM_API_KEY` and `STARTEAM_BASE_URL`, with optional `STARTEAM_MODEL`.

4. Explain the fixed runtime clearly.
   - The LLM plans only operator steps.
   - `clean_data` and `synthesize_data` are the default registered operators.
   - `train_llamafactory` and `eval_model` are fixed runtime stages that run after the planned operators.

5. Point users to durable outputs instead of guessing from memory.
   - Sandbox artifacts live under `workspace/sandboxes/<exp-id>/`.
   - Archived artifacts live under `workspace/artifacts/`.
   - The first files to inspect are `plan.json`, `report.json`, `review.json`, and the step logs under `logs/`.

## Common Responses

- For "how do I run it locally?", give the install command, a minimal `run_task(...)` example, and the expected output files.
- For "why did my run fail?", inspect `report.json`, `review.json`, and the relevant step log before proposing changes.
- For "what should the task file contain?", show the required `TaskTemplate` fields or point at `dataos/templates/*.json`.
- For "what dataset columns do I need?", answer separately for training and evaluation. Training currently requires `instruction` and `response`; evaluation requirements depend on `eval_config.task.name`.

## Boundaries

- Keep runtime execution local and deterministic.
- Do not describe prompts as if they execute work.
- Do not propose new dataset storage formats or new artifact types unless the user explicitly asks to change the runtime contract.
