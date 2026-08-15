# PostTrainBench Fixed-Protocol Alignment Results

Updated: 2026-05-20 18:20 Asia/Shanghai.

This document records the accepted PostTrainBench alignment state for the DataOS
`lm_eval_native` evaluation path. The intended chain is:

`DataOS eval config -> protocol.json -> lm-evaluation-harness task -> vLLM model
generation -> scorer or OpenAI-compatible judge -> metrics.json/summary.json`.

All seven PostTrainBench tasks below have completed eight-model runs through
this chain, with protocol and result artifacts written per model/task. Large
raw generations, judge logs, and remote run directories are intentionally not
committed to the repository; evidence paths are recorded so the runs can be
audited on the experiment server.

Important caveats:

- HealthBench and ArenaHard-Writing use DashScope/Bailian `qwen3.5-plus` as the
  fixed project judge. Official PostTrainBench uses `gpt-5-mini`, so these two
  are protocol-aligned project results rather than strict official-judge
  reproductions.
- AIME 2025 uses the DataOS fixed K32 multi-sample protocol and should not be
  mixed with single-sample AIME scores.
- HumanEval uses the DataOS robust extraction scorer. The strict InspectAI
  `find_code` scorer remains available only for diagnostics because it
  misparses valid chat/thinking/code-block outputs in the direct-vLLM path.
- BFCL uses `max_connections=1` in the DataOS OpenAI-server/lm-eval path. This
  is a documented stability workaround after c6 diagnostics, not an upstream
  PostTrainBench default.

## Fixed Protocol Summary

| Bench | DataOS runner | Core fixed setting | Scorer or judge |
| --- | --- | --- | --- |
| GSM8K | `gsm8k_vllm_native` | 10-shot train examples, seed 42, `max_gen_toks=4000` | Numeric answer match |
| BFCL | `dataos_lmeval_native` with `dataos-local-tool-chat-completions` | Tool-call chat, `max_gen_toks=16000`, `temperature=0.6`, `top_p=0.95`, `max_connections=1` | Strict structured tool-call exact match |
| GPQA Main | `direct_vllm_lmeval` | Frozen historical choice order, `max_gen_toks=16000`, `temperature=1.0`, `max_connections=6` | Final `ANSWER: A/B/C/D` match |
| HumanEval | `direct_vllm_lmeval` | InspectAI/PostTrainBench prompt, `max_gen_toks=4000`, `temperature=1.0`, `max_connections=1` | Robust pass@1 code execution |
| HealthBench | `dataos_lmeval_native` | Official chat payload, `max_gen_toks=16384`, shuffled local JSONL, judge retries/backoff | DashScope `qwen3.5-plus` rubric judge |
| AIME 2025 | `dataos_lmeval_native` | Official InspectAI prompt, K32 repeats, `max_gen_toks=16000`, `temperature=0.6`, `top_p=0.95`, seed omitted | Numeric answer match averaged over K32 |
| ArenaHard-Writing | `dataos_lmeval_native` | Official chat payload, `max_gen_toks=16384`, shuffled local JSONL, official pairwise prompt/baseline policy | DashScope `qwen3.5-plus` pairwise judge |

## Final Eight-Model Results

All values use the task's native percentage/score scale.

| Model | Group | GSM8K | BFCL | GPQA Main | HumanEval | HealthBench | AIME 2025 | ArenaHard-Writing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B-Base | Base | 11.37 | 0.00 | 15.848 | 8.54 | 6.157 | 2.708 | 1.200 |
| Qwen3-4B-Base | Base | 43.21 | 0.00 | 8.705 | 34.76 | 12.826 | 8.438 | 4.600 |
| SmolLM3-3B-Base | Base | 26.31 | 0.00 | 5.134 | 12.20 | 5.777 | 1.042 | 1.155 |
| gemma-3-4b-pt | Base | 3.41 | 9.00 | 0.670 | 0.00 | 5.028 | 0.000 | 0.200 |
| Qwen3-1.7B | Instruct | 88.93 | 93.00 | 38.170 | 69.51 | 34.537 | 33.229 | 38.300 |
| Qwen3-4B | Instruct | 93.56 | 94.00 | 48.661 | 80.49 | 47.651 | 53.229 | 74.300 |
| SmolLM3-3B | Instruct | 80.21 | 85.00 | 35.714 | 73.78 | 23.624 | 35.938 | 21.300 |
| gemma-3-4b-it | Instruct | 83.02 | 66.00 | 25.446 | 72.56 | 43.781 | 12.292 | 90.100 |

ArenaHard-Writing `SmolLM3-3B-Base` was recomputed with the accepted safety-skip
policy after DashScope returned content-inspection failures for three judge
events. The scorer skips only `data_inspection_failed` / inappropriate-content
judge events and still treats 401, 429, quota, billing, and rate-limit errors as
fatal. The final `SmolLM3-3B-Base` Arena score uses 497 successful judge events
and 3 skipped judge events.

## Group Averages vs PostTrainBench Table

This table compares the latest DataOS + lm-eval fixed-protocol averages against
the PostTrainBench reported four-model group averages. Units are percentage
points / task score points, matching the PostTrainBench table convention.

| Bench | Group | DataOS avg | PTB table avg | Delta | Abs delta | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GSM8K | Base | 21.075 | 20.433 | +0.642 | 0.642 | DataOS/lm-eval/vLLM numeric accuracy. |
| GSM8K | Instruct | 86.430 | 86.998 | -0.567 | 0.567 | DataOS/lm-eval/vLLM numeric accuracy. |
| BFCL | Base | 2.250 | 1.500 | +0.750 | 0.750 | DataOS fixed c1 strict tool-call protocol. |
| BFCL | Instruct | 84.500 | 85.000 | -0.500 | 0.500 | DataOS fixed c1 strict tool-call protocol. |
| GPQA Main | Base | 7.589 | 8.480 | -0.891 | 0.891 | Frozen choice order fixed protocol. |
| GPQA Main | Instruct | 36.998 | 36.215 | +0.783 | 0.783 | Frozen choice order fixed protocol. |
| HumanEval | Base | 13.875 | 12.807 | +1.068 | 1.068 | Robust extraction scorer for DataOS/lm-eval. |
| HumanEval | Instruct | 74.085 | 71.493 | +2.593 | 2.593 | Robust extraction scorer for DataOS/lm-eval. |
| HealthBench | Base | 7.447 | 9.490 | -2.043 | 2.043 | DashScope `qwen3.5-plus` judge, not official `gpt-5-mini`. |
| HealthBench | Instruct | 37.398 | 43.320 | -5.922 | 5.922 | DashScope `qwen3.5-plus` judge, not official `gpt-5-mini`. |
| AIME 2025 | Base | 3.047 | 1.665 | +1.382 | 1.382 | DataOS fixed K32 multi-sample protocol. |
| AIME 2025 | Instruct | 33.672 | 29.168 | +4.504 | 4.504 | DataOS fixed K32 multi-sample protocol. |
| ArenaHard-Writing | Base | 1.789 | 1.260 | +0.529 | 0.529 | DashScope `qwen3.5-plus` judge; 3 safety-skip judge events for SmolLM3-Base. |
| ArenaHard-Writing | Instruct | 56.000 | 70.210 | -14.210 | 14.210 | DashScope `qwen3.5-plus` judge; judge-model sensitivity is expected. |

| Scope | Mean absolute delta |
| --- | ---: |
| Base groups | 1.043 |
| Instruct groups | 4.154 |
| All 14 groups | 2.599 |

The largest gaps are ArenaHard-Writing instruct, HealthBench instruct, AIME
instruct, and HumanEval instruct. The first two are expected to be especially
judge-model-sensitive because DataOS uses `qwen3.5-plus` while the PTB table
uses `gpt-5-mini`.

## Evidence Paths

| Bench | Accepted evidence |
| --- | --- |
| GSM8K | Prior full base/instruct reports under `reports/dataos_posttrainbench_gsm8k_four_base_2026-05-11.html` and `reports/dataos_posttrainbench_gsm8k_four_instruct_2026-05-11.html`. |
| BFCL | `/home/hlv/dataos_lmeval_matrix_runs_20260514/bfcl_temp06_strict_sanity_bsz1_20260517_1856`; `/home/hlv/dataos_lmeval_matrix_runs_20260514/bfcl_temp06_strict_remaining_20260517_1900`. |
| GPQA Main | `/home/hlv/dataos_lmeval_matrix_runs_20260514/gpqamain_fixed_protocol_full_20260518_093031`. |
| HumanEval | `/home/hlv/dataos_lmeval_matrix_runs_20260514/humaneval_fixed_protocol_full_20260517_2235`. |
| HealthBench | `/home/hlv/dataos_lmeval_matrix_runs_20260514/healthbench_official_payload_full8_20260518_1408`. |
| AIME 2025 | `/home/hlv/dataos_lmeval_matrix_runs_20260514/aime2025_fixed_k32_full8_20260518_134711`. |
| ArenaHard-Writing | `/home/hlv/dataos_lmeval_matrix_runs_20260514/arena_official_payload_full8_20260519_073202`; `SmolLM3-3B-Base/arenahardwriting` includes the safety-skip rerun summary. |

## Fixed Training Pool Update

This PR also carries the PostTrainBench fixed training-pool manifest used by
the DataOS-controlled agent experiments. The manifest is a source registry, not
a capped merged dataset: agents may select sources, rows, filters, mixture
weights, and schedules only inside this registry.

- Pool id: `posttrainbench_fixed_pool_v1_full`.
- Source count: 24.
- Total processed rows recorded by the manifest: 17,168,338.
- ShareGPT view: `posttrainbench_fixed_pool_v1_full_sharegpt`.
- Main rule: no benchmark evaluation data, no replacement-answer benchmark
  questions, and no additional training datasets outside the registry for the
  main setting.

The previous 16-source fixed pool had already covered the broad math,
tool-use, medical, writing, science, and code areas. On 2026-05-20 the server
manifest was extended to 24 sources to increase coverage for the two thinnest
areas used by the PostTrainBench fixed-pool experiments: GPQA-style science
reasoning and HumanEval-style code generation.

2026-05-20 fixed-pool extension:

| Area | Added/recorded fixed-pool sources | Rows |
| --- | --- | ---: |
| GPQA / science reasoning | `chempile_instruction_reasoning`, `megascience_textbook_reasoning`, `openbookqa_train`, `qasc_train`, `sciq_train` | 727,925 |
| HumanEval / code | `codesearchnet_python_train`, `magicoder_evol_instruct_110k`, `taco_train` | 542,850 |

Current full fixed-pool coverage:

| Area | Fixed-pool sources | Rows |
| --- | --- | ---: |
| GSM8K + AIME / math reasoning | `numinamath_cot`, `openmathinstruct2_train` | 14,832,285 |
| BFCL / tool use | `glaive_function_calling_v2`, `xlam_function_calling_60k` | 172,960 |
| GPQA / science reasoning | `ai2_arc_challenge_train`, `ai2_arc_easy_train`, `chempile_instruction_reasoning`, `megascience_textbook_reasoning`, `openbookqa_train`, `qasc_train`, `scienceqa_train`, `sciq_train`, `sciriff_train_mix` | 814,726 |
| HumanEval / code | `apps_train`, `codesearchnet_python_train`, `magicoder_evol_instruct_110k`, `magicoder_oss_instruct_75k`, `taco_train` | 623,047 |
| HealthBench / medical QA | `medmcqa_train`, `pubmedqa_artificial`, `pubmedqa_labeled`, `pubmedqa_unlabeled` | 456,340 |
| ArenaHard-Writing / instruction writing | `ultrachat_200k_sft`, `ultrafeedback_binarized_sft` | 268,980 |

The ShareGPT export script preserves provenance columns so later training runs
can still audit `source_id`, source repo/config/split, domain tags, benchmark
tags, license, and contamination notes. Both the processed-source registry and
the ShareGPT view were updated on the H100 side and synchronized into
the fixed-pool resource area; the ShareGPT extension export log is
`<pools_root>/logs/sharegpt_export_extension_20260520.log`.
