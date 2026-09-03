# Curation-Bench task adapter

The adapter keeps the benchmark implementation outside DataOS and pins it as an environment
provider. The three task contracts are the only driver selection points:

- `task.yaml`: one run-scoped official N-iteration suite plus multimodal curation in an
  experiment-local harness worktree;
- `train/task.yaml`: `dataos-train task-submit` schedules the official
  `audit -> submit -> finetune` workflow;
- `eval/task.yaml`: `dataos-eval` runs candidate inference on the offline GPU node, runs the
  official VLMEvalKit judge through the environment gateway on the login node, and applies the
  pinned benchmark's eight declared score selectors.

PostTrainBench has its own three explicit contracts, so adding this task does not change its
Operator/dataos-train/lm-eval path. Directory existence has no runtime meaning.

## Deployment requirements

Add one `tools` provider to the selected environment with `root`, `path`, `profile`, `provides`,
`distribution`, `version`, and `revision`; see the example environments. The provider checkout and
CLI venv must match all pins. The official harness's two vendor venvs must also be built at:

- `<root>/vendor/curation-train/.venv`
- `<root>/vendor/VLMEvalKit/.venv`

Build each vendor project with its own `UV_PROJECT_ENVIRONMENT`; never install those dependencies
into a DataOS venv. The external profile must map the selected dataset/model IDs to the same
absolute `inputs.fixed_pool` / `inputs.base_model_path` values used by the run config, name the
evaluation data directory, and configure a gateway reachable from the login node. Judge
credentials remain in the environment's `0600` key file and never enter a Slurm job.

The pinned upstream revision currently has no usable `qwen2-vl-2b` model registry entry even
though it ships a task with that key. Treat
`llava665k_qwen2vl2b_8bench_10k_unlimited` as an upstream preflight failure until the provider pin
is upgraded; the SmolVLM task is the smallest first end-to-end validation target.

The DataOS optimization tree starts with the official seeded random-curation baseline. The
upstream protocol's separate base-model pre-init measurement is informational and is not a DataOS
tree node in this adapter. `budget.iterations` is copied into the official suite manifest and is
also DataOS's stop condition. The random baseline consumes iteration 1; every later valid tree
node claims exactly one subsequent target. Planning errors before a target is activated consume
nothing, while a failed activated target consumes its iteration as required by the harness.
