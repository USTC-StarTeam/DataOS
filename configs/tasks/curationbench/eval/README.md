# EVAL — score a model (external harness)

curationbench's evaluator is the external VLMEvalKit inside the curation-bench provider selected
by `env.yaml` (not vendored here — ~1TB with datasets/weights). EVAL takes the model
produced by the frozen TRAIN stage and scores it on the 8 VLM benchmarks, exactly like
posttrainbench's evaluator scores a model — same shape (`model → score`).

`task.yaml` explicitly selects the external driver and the pinned official executable. The
task-local `collect.py` reads the official `eval/results/results.json`, requires all eight
declared scores, applies the declared maxima, and writes DataOS's canonical `summary.json`.

## Runtime failure handling

The model adapter is resolved from the active task through the pinned harness registry.
Qwen2.5-VL-3B uses `vlmeval_entry.py` to disable vLLM's custom all-reduce kernel and use
standard NCCL while retaining TP=2 and the official model/decoding configuration. An engine
initialization exception terminates the job instead of being retried for every benchmark.
The official vendor checkout and training recipe are not patched.

Upstream VLMEvalKit can exit zero even when every benchmark raised an exception. Therefore
`run_vlmeval.py` checks full prediction-index coverage, nonempty answers and absence of failure
sentinels for every benchmark. These checks run after inference and before calling the judge.
After judging, every declared numeric score is validated before the suite target is completed.
An empty results directory or partial result is a failure, never a completed iteration.

Iteration-budget tasks stop with `execution:invalid_result` when an experiment fails or returns
no finite score. `execution_blocked.json` records the node and report. The loop does not ask for
another strategy, create a review, activate the next official target or refund spent iterations.
Repair and reconcile the current report/target before explicitly resuming; an unchanged failed
report stops again on the same node. Time-budget PTB behavior is unchanged.
