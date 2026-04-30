---
name: debug-dataos-run
description: Diagnose failing, incomplete, or confusing DataOS runs by inspecting sandbox files, experiment reports, plan and review JSON, runtime step records, and archived artifacts in this repository. Use when asked why a run failed, why outputs look wrong, why a step was skipped, how to interpret `report.json` or `review.json`, or how to debug DataOS from a workspace sandbox.
---

# Debug DataOS Run

Use this skill when the problem is about understanding a concrete run outcome, not designing a new feature.

## Follow This Order

1. Identify the run root first.
   - Prefer an explicit sandbox path under `workspace/sandboxes/<exp-id>/`.
   - If only the workspace is known, find the relevant experiment under `workspace/sandboxes/` and archived experiment artifact under `workspace/artifacts/experiments/`.
   - Then read [debug-playbook.md](references/debug-playbook.md).

2. Read the high-signal files in a fixed order.
   - `report.json`
   - `plan.json`
   - `review.json`
   - `logs/*.log`
   - artifact `meta.json` files for the inputs and outputs of the failing step

3. Use `report.json` as the execution truth.
   - Check `status`, `error`, and `steps`.
   - For each step, use `step_type` to distinguish planned operators from fixed runtime stages.
   - Trust the recorded `params`, input artifact ids, and output artifact ids over assumptions.

4. Branch the investigation by failure class.
   - Plan failure or odd plan: inspect `plan.json` and fallback behavior in `dataos/core/agent.py`.
   - Operator failure: inspect the failed operator step, its input dataset artifact, and the matching operator log.
   - Training failure: inspect the `train_llamafactory` runtime step and `logs/train_llamafactory.log`.
   - Eval failure: inspect the `eval_model` runtime step, `logs/eval_model.log`, and eval artifact traces if they were persisted.
   - Missing review or odd review text: inspect the fallback behavior in `LLMPlanningAgent.review()`.

5. Compare artifacts, not just messages.
   - Verify whether the expected output artifact directory exists.
   - Read `meta.json` to confirm parent lineage, splits, row counts, and source ids.
   - For dataset issues, check whether the latest `DatasetArtifact` metadata matches what the next stage expected.

6. Prefer concrete fixes.
   - If the failure is configuration, point to the exact task JSON key or env var.
   - If the failure is code, point to the exact module and step boundary.
   - If the failure is an unrelated worktree issue, say so explicitly instead of attributing it to the run.

## Common Diagnoses

- Missing `STARTEAM_API_KEY` or `STARTEAM_BASE_URL` when no custom `llm_client` is passed.
- Input dataset path missing or not a Hugging Face `save_to_disk` directory.
- Planned operator produced a dataset whose columns no longer match the training or eval stage expectations.
- `train_llamafactory` config missing required fields for `llamafactory` mode.
- `eval_model` split mismatch, missing model artifact, or unsupported eval task config.

## Boundaries

- Do not start by changing the runtime contract or task schema.
- Do not trust README prose over the current code path when they disagree.
- Distinguish failures in the run itself from unrelated syntax errors or dirty worktree issues already present in the repo.
