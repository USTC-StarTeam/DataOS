# DataOS Run Debug Playbook

## File Order

When debugging a run, inspect files in this order:

1. `report.json`
2. `plan.json`
3. `review.json`
4. `logs/*.log`
5. relevant artifact directories and `meta.json`

`report.json` is the execution source of truth because it records:

- experiment status
- experiment error
- all step records
- artifact lineage
- timestamps

## Sandbox Layout

Each sandbox is expected to contain:

- `inputs/`
- `outputs/`
- `artifacts/`
- `logs/`
- `tmp/`
- `code/`
- `plan.json`
- `report.json`
- `review.json`

Archived experiment artifacts are copied into `workspace/artifacts/experiments/`.

## Step Interpretation

Each step record in `report.json` includes:

- `step_type`
- `step_name`
- `input_artifact_ids`
- `output_artifact_ids`
- `params`
- `status`
- `error`

Interpret them like this:

- `step_type == "operator"`
  - planned by the LLM or fallback plan
  - executed through the operator registry

- `step_type == "runtime"`
  - fixed runtime stage
  - currently `train_llamafactory` or `eval_model`

## Fast Triage By Symptom

### Run fails before or around planning

Check:

- whether `plan.json` exists
- whether `_build_client_from_env()` requirements are satisfied when no custom `llm_client` was provided
- whether the initial dataset path exists

Likely causes:

- missing `STARTEAM_API_KEY`
- missing `STARTEAM_BASE_URL`
- invalid dataset input path

### Operator step fails

Check:

- the failed step entry in `report.json`
- the operator-specific log in `logs/`
- input and output dataset artifact `meta.json`

Likely causes:

- missing required dataset artifact in `inputs`
- incompatible dataset fields
- operator logic raised a validation error

### Training step fails

Check:

- runtime step `train_llamafactory` in `report.json`
- `logs/train_llamafactory.log`
- model artifact directory
- `training_config.json` or `training_manifest.json` when present

Likely causes:

- invalid `train_config`
- missing LLaMA-Factory binary or environment
- dataset missing required `instruction` / `response` columns
- unsupported train override keys

### Eval step fails

Check:

- runtime step `eval_model` in `report.json`
- `logs/eval_model.log`
- eval artifact or result trace artifacts
- dataset split selection and model artifact lineage

Likely causes:

- missing `ModelArtifact`
- requested split not found
- unsupported eval task or adapter
- invalid `eval_config`
- benchmark dataset missing required fields

### Review output looks wrong

Check:

- whether `review.json` contains fallback wording
- `LLMPlanningAgent.review()` fallback branch in `dataos/core/agent.py`

Likely cause:

- review generation failed and the runtime wrote a fallback review

## Useful Questions To Answer

When summarizing the diagnosis, answer these explicitly:

1. Which step failed?
2. Was it an operator step or a runtime stage?
3. What exact file or config proves the failure mode?
4. Is the issue in task config, dataset contents, environment, or code?
5. What is the smallest next fix?
