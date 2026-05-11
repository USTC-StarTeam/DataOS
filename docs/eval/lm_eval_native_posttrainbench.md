# lm-eval Native Backend and PostTrainBench Protocol Notes

This note records the rationale and operating contract for the DataOS
evaluation refactor that adds lm-evaluation-harness native execution and the
first PostTrainBench protocol package.

## Why This Change Exists

DataOS originally evaluated models through the DataOSLM adapter bridge. In that
path, DataOS owns prompt rendering and model inference, then exposes a model-like
interface to lm-evaluation-harness. That remains useful for DataOS-native
runtime experiments, but it is not enough for benchmark alignment work where the
benchmark ecosystem already expects lm-eval's own model backends.

For PostTrainBench alignment, the desired contract is different:

1. DataOS freezes and records the experiment protocol.
2. lm-evaluation-harness loads the task definition, owns model execution through
   backends such as `hf` or `vllm`, and computes the metric.
3. DataOS stores both the result and the resolved protocol so a run can be
   audited later.

This PR adds that second path as `mode="lm_eval_native"` while keeping the
legacy adapter mode intact.

## What DataOS Owns

DataOS should be the protocol owner. A protocol is more than a task name. It
includes:

- task YAML and custom metric functions;
- dataset name, dataset split, few-shot policy, and random seeds;
- system instruction and prompt construction;
- chat template selection for model families when the upstream benchmark uses
  model-specific chat formatting;
- generation arguments, stop sequences, and max generation length;
- lm-eval backend and backend-specific model arguments;
- source model path or checkpoint identity.

The native backend writes a `protocol.json` file next to `results.json` and
`lm_eval_results.json`. This file captures the user-provided `EvalConfig`, the
source `ModelArtifact`, and the resolved lm-eval model settings.

## What lm-eval Owns

lm-evaluation-harness remains responsible for task loading, request
construction, model backend execution, sample logging, and metric aggregation.
DataOS does not reimplement those pieces in the native path. It passes a frozen
configuration to lm-eval and records the raw lm-eval output for review.

## Current PostTrainBench Scope

The initial PostTrainBench package contains only GSM8K:

- dataset: `openai/gsm8k`, `main`, `test`;
- few-shot policy: 10 train examples shuffled with seed 42 and passed as the
  system instruction;
- model-specific chat templates for Qwen3, SmolLM3, and Gemma3;
- metric: numeric end-match accuracy compatible with the Inspect AI
  `match(numeric=True)` behavior used by PostTrainBench;
- generation: `max_gen_toks=4000` in the verification runner.

This is not full PostTrainBench coverage. New PostTrainBench tasks should be
added one protocol at a time with their own YAML, helper functions, template
requirements, reference results, and validation notes.

## Validation Evidence

The native path was validated on server `5090-1` with local model weights and
the vLLM backend.

### Base GSM8K

| Model | DataOS | PostTrainBench official/reference | Delta |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B-Base | 11.37 | 12.66 | -1.29 |
| Qwen3-4B-Base | 43.21 | 41.85 | +1.36 |
| SmolLM3-3B-Base | 26.31 | 21.08 | +5.23 |
| gemma-3-4b-pt | 3.41 | 6.14 | -2.73 |
| Average | 21.08 | 20.43 | +0.65 |

### Instruct GSM8K

| Model | DataOS | PostTrainBench current reference | Delta |
| --- | ---: | ---: | ---: |
| Qwen3-1.7B | 88.93 | 88.63 | +0.30 |
| Qwen3-4B | 93.56 | 93.33 | +0.23 |
| SmolLM3-3B | 80.21 | 83.24 | -3.03 |
| gemma-3-4b-it | 83.02 | 84.69 | -1.67 |
| Average | 86.43 | 87.47 | -1.04 |

These numbers support the narrower claim that the DataOS -> lm-eval native ->
vLLM -> metric chain does not have an obvious systematic score collapse. They do
not prove complete PostTrainBench reproduction. The remaining deltas should be
treated as protocol-alignment work, especially around generation settings, chat
templates, model revisions, and sample-level differences.

## Review Guidance

When reviewing or extending this area, avoid claims such as "DataOS fully
supports PostTrainBench." The current claim is:

> DataOS supports lm-eval native execution and includes an initial,
> auditable PostTrainBench GSM8K protocol package validated on real base and
> instruct models.

