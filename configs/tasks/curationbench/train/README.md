# TRAIN — frozen

This directory's **existence signals that training is frozen** for curationbench: the agent
does NOT choose the recipe/hyperparameters. The brain derives the train stage's ownership
(`provided — frozen`) from the presence of this dir (see `compose.resolve`).

The actual finetune runs the external curation-bench harness at `run.bench_root`
(`datacuration-bench audit → submit → finetune`, using its vendored `curation-train` with a
fixed recipe). It consumes the agent's curated subset and produces a model at the harness
`run_dir/finetune/`, which the EVAL stage then scores. See `train.yaml` for the command.

Contrast: posttrainbench has NO `train/` dir → its training is the agent's own method
(dataos-train, free hyperparameters).
