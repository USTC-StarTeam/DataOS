# lm-eval Native Backend and PostTrainBench Protocol Notes

> ⚠️ **Partly superseded (post-2026-06 agent-version refactor)** — the core rationale below
> (lm-eval owns model execution via `vllm`/`hf`; DataOS owns and freezes the protocol) still
> holds, but the **config-API specifics are stale**: the `mode="lm_eval_native"` /
> `EvalConfig` / DataOSLM "legacy adapter" path is **removed**. Eval is now the self-contained
> **`configs/tasks/posttrainbench/eval/`** (run by the sub-agent on an H100 Slurm node
> via `run_lmeval_eval.py`). Server-backend concurrency is now a uniform default of **32** with a
> `--num-concurrent` override (was a per-bench 1/2/6); BFCL also extracts bare-JSON tool calls
> (`_extract_embedded_json`). Current truth: **AGENTS.md / README.md**.

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

## Protocol Hygiene Rule

PostTrainBench paper runs must use one fixed scoring protocol per benchmark.
Score-affecting generation settings must not depend on whether a checkpoint is
base, pre-trained, or instruction-tuned, and must not be inferred from a
checkpoint's `generation_config.json`.

Allowed model-dependent behavior is limited to formatting and serving details:

- model-family chat templates, matching the upstream PostTrainBench evaluators;
- BFCL tool parser selection needed by vLLM to expose structured tool calls;
- resource-only settings such as GPU memory utilization.

The protocol runners therefore pass explicit per-task generation settings. They
do not read `generation_config.json` from the evaluated checkpoint. Any
diagnostic override such as `--temperature` or `--do-sample` must be recorded as
an override and must not be mixed into final paper tables unless the same value
is applied to every model for that benchmark.

## Accepted Results Snapshot

The accepted seven-task, eight-model DataOS fixed-protocol results are recorded
in [posttrainbench_fixed_protocol_results_2026-05-20.md](posttrainbench_fixed_protocol_results_2026-05-20.md).
That document contains:

- the final per-model scores for GSM8K, BFCL, GPQA Main, HumanEval,
  HealthBench, AIME 2025, and ArenaHard-Writing;
- four-base-model and four-instruct-model averages compared against the
  PostTrainBench reported table averages;
- evidence paths for each accepted server-side run;
- the fixed training-pool expansion summary used by the DataOS controlled
  PostTrainBench experiments, including the 24-source / 17,168,338-row
  2026-05-20 manifest and its ShareGPT view.

This note remains the implementation and protocol contract. The results
snapshot is the paper-table comparison record.

### HumanEval Scorer Decision

HumanEval is aligned as a DataOS -> lm-eval native -> direct vLLM protocol with a
DataOS robust code-extraction scorer. The protocol target remains the
PostTrainBench/InspectAI HumanEval target: execute the HumanEval prompt,
extracted candidate code, test body, and `check(entry_point)` for pass@1. The
DataOS scorer intentionally normalizes chat-model output before execution:

- strip `<think>...</think>` reasoning blocks;
- prefer fenced Python code when present;
- remove repeated function signatures/prompts;
- normalize generated Python artifacts such as Gemma sentencepiece spaces.

This robust scorer is the accepted default for DataOS/lm-eval HumanEval runs.
The strict InspectAI `find_code` extraction is retained only as a diagnostic,
not as the default scorer. Offline strict rescoring of the accepted 2026-05-17
eight-model run showed large, shape-driven collapses: Qwen3-4B dropped from
80.49 to 28.05 and gemma-3-4b-it dropped from 72.56 to 0.00. That behavior is
not acceptable for the lm-eval direct-vLLM chain because it misparses valid
chat/thinking/code-block outputs rather than revealing a generation protocol
difference.

Accepted HumanEval evidence:

- report: `reports/posttrainbench_humaneval_fixed_protocol_alignment_2026-05-17.html`
- remote run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/humaneval_fixed_protocol_full_20260517_2235`
- strict diagnostic:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/humaneval_fixed_protocol_full_20260517_2235/strict_official_find_code_rescore.jsonl`

### BFCL Protocol Decision

BFCL is aligned as a DataOS -> lm-eval native -> OpenAI-compatible vLLM tool
calling protocol. The accepted fixed protocol uses one generation setting for
all base and instruction-tuned checkpoints:

- `max_gen_toks=16000`
- `temperature=0.6`
- `top_p=0.95`
- `batch_size=1`
- `max_connections=1`
- `--generation-config vllm`
- vLLM `--enable-auto-tool-choice`
- upstream model-family tool parser (`hermes` for Qwen, Gemma, and SmolLM)

DataOS registers `dataos-local-tool-chat-completions` because lm-eval's stock
local chat-completions backend returns only message text and drops structured
OpenAI `tool_calls`. The DataOS backend serializes structured `tool_calls` so
the task scorer can apply the same exact function/argument matching target as
PostTrainBench BFCL.

The BFCL scorer is intentionally strict. It accepts structured OpenAI
`tool_calls` and explicit tool-call blocks/tags, but does not treat generic JSON
code blocks inside explanatory prose as tool calls. That avoids false positives
from base models that describe a possible function call without actually making
one.

The `max_connections=1` setting is a DataOS/lm-eval path workaround, not a
claim of upstream equivalence. Upstream PostTrainBench BFCL defaults to
`max_connections=6`; diagnostics showed official InspectAI c6 runs succeed
while the DataOS OpenAI-server/lm-eval c6 path collapses tool-call scoring.
Result tables should therefore mention the c1 caveat.

Accepted BFCL evidence:

- report: `reports/posttrainbench_bfcl_fixed_protocol_alignment_2026-05-17.html`
- sanity run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/bfcl_temp06_strict_sanity_bsz1_20260517_1856`
- remaining run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/bfcl_temp06_strict_remaining_20260517_1900`
- c6 diagnostics:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/bfcl_alignment_sanity_fixedgen_20260517_2345`
  and
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/bfcl_alignment_sanity_parserfinal_20260517_2350`

### GPQA Main Protocol Decision

GPQA Main is aligned as a DataOS -> lm-eval native -> direct vLLM protocol. The
accepted fixed protocol uses one generation setting for all base and instruction
tuned checkpoints:

- `max_gen_toks=16000`
- `temperature=1.0`
- stop sequences `</s>`, `Question:`, `<|im_end|>`, and `<|eot_id|>`
- `max_connections=6`
- `--generation-config vllm`
- `num_fewshot=0`
- model-family chat templates from the PostTrainBench evaluator

Upstream PostTrainBench does not explicitly pass `temperature`, `top_p`, or
`top_k` in `src/eval/tasks/gpqamain/evaluate.py`. InspectAI's `GenerateConfig`
defaults those fields to `None`, and official historical InspectAI logs record
only `max_connections=6` and `max_tokens=16000` in the first model event.
However, vLLM's default `SamplingParams` has `temperature=1.0`, `top_p=1.0`, and
`top_k=0`. DataOS therefore pins `temperature=1.0` and `generation_config=vllm`
in the protocol so final runs are not affected by per-checkpoint
`generation_config.json` files.

Choice order is the other important GPQA reproducibility issue. Upstream
PostTrainBench uses InspectAI `hf_dataset(..., shuffle_choices=True)`, and
historical logs show different choice orders across separate official-style
evaluator runs. DataOS freezes a single choice order for final tables. The frozen
order comes from the historical InspectAI log:

`/home/hlv/dataos_lmeval_verify_20260511_140721/logs/2026-05-12T02-54-11+00-00_gpqa-main_WggnDvnBSikgdmQc23ijus.json`

and is provided at runtime as:

`/home/hlv/posttrainbench_benchmarks/local/gpqa_main_choice_order_20260512.jsonl`

This is a deliberate paper-protocol choice. It improves reproducibility but means
exact agreement with every historical PostTrainBench/5090 run is not expected
when those historical runs used different shuffled choice orders.

Accepted GPQA evidence:

- report: `reports/posttrainbench_gpqa_fixed_protocol_alignment_2026-05-18.html`
- sanity run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/gpqamain_fixed_protocol_sanity_20260518_092325`
- full run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/gpqamain_fixed_protocol_full_20260518_093031`

### HealthBench Protocol Decision

HealthBench is wired as a DataOS -> lm-eval native -> OpenAI-compatible vLLM ->
LLM-as-judge protocol. It is a fixed project protocol, not a strict numerical
replication of upstream PostTrainBench scores, because the judge differs:
upstream uses `gpt-5-mini`, while DataOS paper runs use DashScope/Bailian
`qwen3.5-plus`.

The fixed DataOS protocol uses:

- local PostTrainBench HealthBench JSONL, shuffled with `random.Random(42)`
- model-family chat templates from the PostTrainBench evaluator
- `max_gen_toks=16384`
- vLLM `--generation-config vllm`
- `dataos_official_chat_payload=true`, meaning the DataOS lm-eval backend sends
  only `model`, `messages`, and `max_tokens` to `/v1/chat/completions` for
  HealthBench. This matches the official evaluator and avoids lm-eval's default
  `temperature=0`, `seed`, and `stop` injection.
- official HealthBench grader prompt and weighted rubric scoring
- `qwen3.5-plus` judge through
  `https://dashscope.aliyuncs.com/compatible-mode/v1`
- non-GPT judge request settings: `temperature=0`, `max_tokens=512`,
  JSON-object response format, and DashScope thinking disabled
- 5 judge retries with 2 second linear backoff
- at most 8 rubric workers per sample and at most 4 concurrent judge requests

The protocol runner records this non-secret judge policy in `protocol.json`, and
the task writes per-rubric evidence to `healthbench_judge_events.jsonl`. API key
values must only come from the environment and must not appear in run artifacts
or documentation.

Current HealthBench evidence:

- report:
  `reports/posttrainbench_healthbench_fixed_protocol_alignment_2026-05-18.html`
- official direct diagnostic:
  `/home/hlv/healthbench_official_direct_diag_20260518`
- fixed official-payload paired sanity:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/healthbench_official_payload_sanity_20260518_1305`
- fixed official-payload four-Qwen limit-32 run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/healthbench_official_payload_qwen4_limit32_20260518_1310`
- old pre-fix diagnostic evidence, retained only as a rejected counterexample:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/healthbench_qwen35_qwen4_limit32_20260518_1150`

## What lm-eval Owns

lm-evaluation-harness remains responsible for task loading, request
construction, model backend execution, sample logging, and metric aggregation.
DataOS does not reimplement those pieces in the native path. It passes a frozen
configuration to lm-eval and records the raw lm-eval output for review.

## Current PostTrainBench Scope

The PostTrainBench package now contains lm-eval task entries for the full
seven-task suite:

- AIME 2025: `posttrainbench_aime2025`
- Arena-Hard Writing: `posttrainbench_arena_hard_writing`
- BFCL: `posttrainbench_bfcl`
- GPQA Main: `posttrainbench_gpqa_main`
- GSM8K: `posttrainbench_gsm8k`
- HealthBench: `posttrainbench_healthbench`
- HumanEval: `posttrainbench_humaneval`
- Suite group: `posttrainbench_suite`

GSM8K, GPQA Main, BFCL, HumanEval, HealthBench, Arena-Hard Writing, and AIME
2025 have completed eight-model fixed-protocol runs on `5090-1` through the
DataOS -> lm-eval native -> vLLM path. HealthBench and Arena-Hard Writing add a
DashScope/Bailian `qwen3.5-plus` OpenAI-compatible judge after local model
generation, so their numbers are fixed project-judge results rather than strict
official `gpt-5-mini` reproductions.

### Coverage Matrix

| PostTrainBench task | Current runner in `run_posttrainbench_protocol_eval.py` | DataOS lm-eval native task asset? | Interpretation |
| --- | --- | --- | --- |
| GSM8K | `gsm8k_vllm_native` | Yes, YAML + helper functions | Eight-model fixed-protocol run aligned; 10-shot seed-42 numeric accuracy. |
| AIME 2025 | `dataos_lmeval_native` | Yes, YAML + helper functions | Eight-model fixed K32 run completed; independent repeats omit the OpenAI-compatible seed. |
| Arena-Hard Writing | `dataos_lmeval_native` | Yes, Python task | Eight-model fixed project-judge run completed with DashScope `qwen3.5-plus`; SmolLM3-Base records three content-safety judge skips. |
| BFCL | `dataos_lmeval_native` | Yes, Python task + DataOS chat backend | Eight-model fixed-protocol run aligned on 2026-05-18; uses strict tool-call parsing with a documented `max_connections=1` caveat. |
| GPQA Main | `direct_vllm_lmeval` | Yes, YAML + helper functions | Fixed full protocol accepted with frozen-choice-order caveat. |
| HealthBench | `dataos_lmeval_native` | Yes, Python task | Eight-model fixed project-judge run completed with DashScope `qwen3.5-plus`; not an exact official-score reproduction because judge differs. |
| HumanEval | `direct_vllm_lmeval` | Yes, YAML + robust code-eval metric | Eight-model fixed-protocol run aligned on 2026-05-17; strict `find_code` is diagnostic only. |

Similarly, the eight base/instruct model paths used during validation are not
currently registered as a DataOS model matrix. They are passed to the protocol
runners at execution time through `--model-path` and `--model-name`.

The model list is still supplied at execution time through `--model-path` and
`--model-name`; it has not yet been moved into a central model matrix.

## Validation Evidence

The native path was validated on server `5090-1` with local model weights and
the vLLM backend.

### Historical Seven-Task lm-eval Smoke Status

These smoke tests used `Qwen3-1.7B-Base` with `limit=1`. They are intended to
validate execution plumbing and artifact creation, not benchmark quality.

| Task | Metric observed | Status |
| --- | ---: | --- |
| GSM8K | `accuracy=0.0` | Passed; artifact and metrics written. |
| GPQA Main | `accuracy=1.0` | Passed; artifact and metrics written. |
| BFCL | `bfcl_accuracy=0.0` | Passed; artifact and metrics written. |
| HumanEval | `pass@1=0.0` | Passed; artifact and metrics written. |
| AIME 2025 | `exact_match@k32=0.0` | Passed; artifact and metrics written. |
| Arena-Hard Writing | `arena_winrate=0.0` | Passed with DashScope `qwen3.5-plus` judge; artifact, metrics, summary, samples, and per-game evidence written. |
| HealthBench | `Qwen3-1.7B-Base limit=1 score=0.0` | Passed with DashScope `qwen3.5-plus` judge; artifact, metrics, summary, samples, and per-rubric evidence written. |

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
aligns with PostTrainBench scores." The current claim is:

> DataOS has a full seven-task PostTrainBench lm-eval task suite and accepted
> eight-model fixed-protocol results for GSM8K, BFCL, GPQA Main, HumanEval,
> HealthBench, AIME 2025, and ArenaHard-Writing. HealthBench and ArenaHard are
> fixed project-judge protocols using DashScope `qwen3.5-plus`, not strict
> official `gpt-5-mini` reproductions. BFCL carries the documented c1
> concurrency caveat, GPQA carries the frozen-choice-order caveat, AIME is the
> DataOS fixed K32 protocol, and HumanEval uses the accepted robust extraction
> scorer.
