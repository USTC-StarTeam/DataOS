# TRAIN — frozen

`task.yaml` explicitly selects `driver: external_cli` and `ownership: frozen`; directory
existence has no meaning. The agent does NOT choose the recipe/hyperparameters.

The actual finetune runs the external curation-bench provider selected by `env.yaml`
(`datacuration-bench audit → submit → finetune`, using its vendored `curation-train` with a
fixed recipe). `dataos-train task-submit` schedules that declared workflow and
`task-collect` returns the official `model_path`. See `task.yaml`.

Contrast: PostTrainBench's own `train/task.yaml` explicitly selects `driver: native` and
`ownership: agent` (dataos-train TrainSpec, free hyperparameters).
