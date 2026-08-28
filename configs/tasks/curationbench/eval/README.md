# EVAL — score a model (external harness)

curationbench's evaluator is the external VLMEvalKit inside the curation-bench provider selected
by `env.yaml` (not vendored here — ~1TB with datasets/weights). EVAL takes the model
produced by the frozen TRAIN stage and scores it on the 8 VLM benchmarks, exactly like
posttrainbench's evaluator scores a model — same shape (`model → score`).

`task.yaml` explicitly selects the external driver and the pinned official executable. The
task-local `collect.py` reads the official `eval/results/results.json`, requires all eight
declared scores, applies the declared maxima, and writes DataOS's canonical `summary.json`.
