# EVAL — score a model (external harness)

curationbench's evaluator is the external VLMEvalKit inside the curation-bench harness at
`run.bench_root` (not vendored here — ~1TB with datasets/weights). EVAL takes the model
produced by the frozen TRAIN stage and scores it on the 8 VLM benchmarks, exactly like
posttrainbench's evaluator scores a model — same shape (`model → score`).

The command + native-result contract + normalization (`aggregate`) are declared in
`../task.yaml` (scoring). The common eval runner (`common/eval_runner/evaluate.py`) runs it
and normalizes the per-benchmark scores into the uniform `score.json`.
