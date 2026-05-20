# PostTrainBench Task Assets

This directory contains DataOS-maintained lm-evaluation-harness task assets for
PostTrainBench protocol reproduction.

The current scope covers the seven PostTrainBench evaluation tasks as lm-eval
task entries:

- `posttrainbench_aime2025`
- `posttrainbench_arena_hard_writing`
- `posttrainbench_bfcl`
- `posttrainbench_gpqa_main`
- `posttrainbench_gsm8k`
- `posttrainbench_healthbench`
- `posttrainbench_humaneval`
- `posttrainbench_suite` as the seven-task group

GSM8K, GPQA Main, BFCL, HumanEval, HealthBench, Arena-Hard Writing, and AIME
2025 have completed eight-model fixed-protocol runs on `5090-1` through the
DataOS -> lm-eval native -> vLLM path. HealthBench and Arena-Hard Writing add a
DashScope/Bailian `qwen3.5-plus` OpenAI-compatible judge after local model
generation; those two are fixed project-judge protocols rather than strict
official `gpt-5-mini` reproductions.

The accepted result snapshot and PostTrainBench table comparison are recorded in
`docs/eval/posttrainbench_fixed_protocol_results_2026-05-20.md`.

Arena-Hard Writing and HealthBench call an OpenAI-compatible judge during
`process_results`, so they require judge credentials. BFCL requires the
DataOS-registered `dataos-local-tool-chat-completions` backend so lm-eval keeps
the structured `tool_calls` returned by vLLM's OpenAI-compatible server.

## Protocol Hygiene

Final PostTrainBench runs must use a single fixed generation protocol per
benchmark. Do not branch on base vs instruct checkpoints, and do not inherit
score-affecting generation settings from a model's `generation_config.json`.

The only model-family-specific behavior that belongs in this package is prompt
formatting and serving compatibility:

- `qwen3.jinja`, `smollm.jinja`, and `gemma3.jinja` mirror the upstream
  PostTrainBench chat-template selection.
- BFCL uses `gemma3_tool_calling.jinja` for Gemma because vLLM tool calling
  needs a tool-call-aware template.
- vLLM resource knobs may vary for stability, but they are not benchmark
  scoring settings.

If a diagnostic run uses a generation override such as `temperature`,
`do_sample`, `top_p`, or `top_k`, it must be recorded in `protocol.json` and
treated as a separate protocol version until the same setting is applied to all
models for that benchmark.

## Current Protocol

`posttrainbench_gsm8k.yaml` defines the GSM8K task used by the original
verification runner.

- Dataset: `openai/gsm8k`
- Dataset name: `main`
- Split: `test`
- Output type: `generate_until`
- Few-shot examples: 10 examples from the train split, shuffled with seed 42
  and inserted through the runner as `system_instruction`
- lm-eval `num_fewshot`: `0`, because the 10 examples are injected explicitly
  to match the PostTrainBench chat protocol
- Metric: `accuracy`
- Metric implementation: `utils.process_results`
- Answer rule: numeric end-match, compatible with PostTrainBench's Inspect AI
  numeric match behavior
- Verification runner max generation length: `max_gen_toks=4000`

The runner selects model-family chat templates from `templates/`:

- `qwen3.jinja`
- `smollm.jinja`
- `gemma3.jinja`

Changing any template, prompt rule, extraction rule, seed, stop sequence, or
generation setting changes the protocol and must be treated as a protocol
version update.

### HumanEval Accepted Protocol

`posttrainbench_humaneval.yaml` is the accepted DataOS/lm-eval HumanEval task.
It should be executed through the direct vLLM native lm-eval runner, not the
OpenAI-server chat-completions path.

- Dataset: `openai/openai_humaneval`
- Split: `test`
- Prompt: the InspectAI/PostTrainBench HumanEval instruction plus the original
  HumanEval function signature and docstring
- lm-eval `num_fewshot`: `0`
- Generation: `max_gen_toks=4000`, task-level `temperature=1.0`
- Connection policy: one model stream at a time for protocol parity with
  PostTrainBench `max_connections=1`
- Chat formatting: model-family templates only (`qwen3.jinja`, `smollm.jinja`,
  `gemma3.jinja`)
- Scorer: robust pass@1 code execution in `utils.humaneval_pass_at_k`

The robust scorer is part of the DataOS/lm-eval HumanEval protocol. It keeps the
same execution target as PostTrainBench/InspectAI, but it first normalizes
lm-eval direct-vLLM chat completions by removing thinking blocks, extracting
fenced code when present, removing repeated prompts/function signatures, and
normalizing generated Python text. This is needed because strict InspectAI
`find_code` was shown to misparse valid chat/thinking/code-block outputs in the
lm-eval chain.

Do not switch the default scorer back to strict `find_code` for final tables.
Strict rescoring is allowed only as a diagnostic. The accepted alignment evidence
is:

- local report:
  `reports/posttrainbench_humaneval_fixed_protocol_alignment_2026-05-17.html`
- remote run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/humaneval_fixed_protocol_full_20260517_2235`
- strict diagnostic:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/humaneval_fixed_protocol_full_20260517_2235/strict_official_find_code_rescore.jsonl`

### GPQA Main Accepted Protocol

`posttrainbench_gpqa_main.yaml` is the accepted DataOS/lm-eval GPQA Main task.
It should be executed through the direct vLLM native lm-eval runner.

- Dataset: `Idavidrein/gpqa`
- Dataset name: `gpqa_main`
- Split: `train`
- Prompt: InspectAI/PostTrainBench `multiple_choice(cot=True)` style instruction
  with four answer choices
- lm-eval `num_fewshot`: `0`
- Generation: `max_gen_toks=16000`, task-level `temperature=1.0`
- Stop sequences: `</s>`, `Question:`, `<|im_end|>`, and `<|eot_id|>`
- Connection policy: `max_connections=6`, matching the PostTrainBench GPQA
  evaluator default
- Serving: lm-eval `vllm` backend with `generation_config=vllm`, so score-
  affecting decoding does not inherit a checkpoint `generation_config.json`
- Chat formatting: model-family templates only (`qwen3.jinja`, `smollm.jinja`,
  `gemma3.jinja`)
- Scorer: extract final `ANSWER: A|B|C|D` and compare against the shuffled target

PostTrainBench uses InspectAI `hf_dataset(..., shuffle_choices=True)`. Historical
logs show that choice order is not stable across separate GPQA evaluator runs, so
DataOS freezes the choice order for final tables instead of depending on a fresh
shuffle per run. The current fixed order comes from the official-style historical
InspectAI log:

`/home/hlv/dataos_lmeval_verify_20260511_140721/logs/2026-05-12T02-54-11+00-00_gpqa-main_WggnDvnBSikgdmQc23ijus.json`

and is materialized on the 5090 server as:

`/home/hlv/posttrainbench_benchmarks/local/gpqa_main_choice_order_20260512.jsonl`

That frozen file is not committed because it contains GPQA benchmark content. A
paper run should provide it via `POSTTRAINBENCH_GPQA_FROZEN_JSONL`; the runner
records both that path and `POSTTRAINBENCH_GPQA_SHUFFLE_SEED` in `protocol.json`.

The accepted alignment evidence is:

- local report:
  `reports/posttrainbench_gpqa_fixed_protocol_alignment_2026-05-18.html`
- remote sanity run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/gpqamain_fixed_protocol_sanity_20260518_092325`
- remote full run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/gpqamain_fixed_protocol_full_20260518_093031`

### BFCL Accepted Protocol

`posttrainbench_bfcl.yaml` is the accepted DataOS/lm-eval BFCL task for the
OpenAI-server tool-call path.

- Dataset: Berkeley Function Calling Leaderboard `exec_simple`
- Split/revision: train split at revision
  `1bf8bbc3c0e35d04d00339c223a3fd653aa195ac`, matching PostTrainBench
- Prompt: BFCL question as a chat user message with OpenAI-compatible tool
  schemas attached to the request
- lm-eval `num_fewshot`: `0`
- Generation: `max_gen_toks=16000`, `temperature=0.6`, `top_p=0.95`
- Stop sequences: `<|im_end|>` and `<|eot_id|>`
- Serving: vLLM OpenAI-compatible chat server with `--generation-config vllm`,
  `--enable-auto-tool-choice`, and the upstream model-family tool parser
  (`hermes` for Qwen, Gemma, and SmolLM)
- Connection policy: `max_connections=1` for the DataOS/lm-eval
  OpenAI-server path
- Scorer: exact single tool-call function and arguments match

The `max_connections=1` setting is a documented DataOS/lm-eval stability
workaround, not an upstream default. Upstream PostTrainBench BFCL defaults to
`max_connections=6`; however, alignment diagnostics showed that official
InspectAI c6 runs succeed while the DataOS OpenAI-server/lm-eval c6 path can
collapse tool-call scoring. Keep this caveat attached to BFCL result tables.

The accepted alignment evidence is:

- local report:
  `reports/posttrainbench_bfcl_fixed_protocol_alignment_2026-05-17.html`
- remote sanity run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/bfcl_temp06_strict_sanity_bsz1_20260517_1856`
- remote remaining run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/bfcl_temp06_strict_remaining_20260517_1900`
- c6 diagnostic evidence:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/bfcl_alignment_sanity_fixedgen_20260517_2345`
  and
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/bfcl_alignment_sanity_parserfinal_20260517_2350`

### AIME 2025 DataOS Fixed K32 Protocol

`posttrainbench_aime2025.yaml` is the DataOS/lm-eval AIME 2025 task. It is
currently treated as a fixed K32 multi-sample protocol, not as a single-sample
official-score protocol.

- Dataset: `math-ai/aime25`
- Split: `test`, 30 questions
- Prompt: the upstream `inspect_evals.aime2025` `USER_PROMPT_TEMPLATE`
  exactly; do not append a trailing `Reasoning:` marker
- lm-eval `num_fewshot`: `0`
- Repeats: `32`, with the `k32` filter retaining all 32 generations per sample
- Generation: `max_gen_toks=16000`, `temperature=0.6`, `top_p=0.95`
- Connection policy: `max_connections=6`, matching the PostTrainBench AIME
  evaluator default
- Serving: vLLM OpenAI-compatible chat server with `--generation-config vllm`
- Chat formatting: model-family templates only (`qwen3.jinja`, `smollm.jinja`,
  `gemma3.jinja`)
- Seed policy: omit the OpenAI-compatible `seed` field for AIME K32 sampling.
  lm-eval's OpenAI-compatible backend otherwise injects the same default seed
  for every repeat, which makes K32 samples non-independent and was shown to
  collapse AIME sanity results.
- Scorer: numeric match after the last boxed answer or final `ANSWER:` line,
  averaged across the 32 retained generations

Upstream PostTrainBench `src/eval/tasks/aime2025/evaluate.py` exposes
`max_tokens=16000` and `max_connections=6`, and delegates the task to
`inspect_evals/aime2025`. It does not explicitly expose K32 in that file. For
DataOS paper runs, K32 is therefore recorded as a project fixed multi-sample
protocol and must not be mixed with single-sample scores.

Rejected diagnostic evidence:

- old DataOS/lm-eval full Qwen3-1.7B-Base sanity:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/aime2025_k32_full_base_sanity_20260518_1738/Qwen3-1.7B-Base/aime2025`
- failure mode: score `0.0`; prompt still used the generic math prompt with a
  trailing `Reasoning:` marker, and all 32 repeats shared the OpenAI-compatible
  backend seed

Current alignment evidence:

- official InspectAI limit-3 diagnostic:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/aime2025_official_inspect_qwen17base_limit3_20260518_2000`
- DataOS/lm-eval official-prompt/omit-seed limit-3 diagnostic:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/aime2025_lmeval_officialprompt_omitseed_qwen17base_limit3_20260518_2012`
- result: official InspectAI path scored `14/96 = 14.58%`; DataOS/lm-eval
  scored `15/96 = 15.625%` on the same three-question sanity slice, with
  non-duplicated samples and `protocol.json` recording `dataos_omit_seed=true`
- accepted eight-model full run:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/aime2025_fixed_k32_full8_20260518_134711`
- accepted Qwen3-1.7B-Base full sanity:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/aime2025_fixedprompt_omitseed_qwen17base_full_20260518_2056/Qwen3-1.7B-Base/aime2025`

### HealthBench DataOS Protocol

`posttrainbench_healthbench.yaml` is the DataOS/lm-eval HealthBench task. It is
accepted as a working DataOS -> lm-eval -> vLLM -> OpenAI-compatible judge
pipeline, but not as a strict numerical reproduction of upstream
PostTrainBench because the judge provider/model intentionally differs.

- Dataset: PostTrainBench local HealthBench JSONL
- Split/order: the official local JSONL shuffled with `random.Random(42)`
- Prompt: the sample conversation is sent as chat messages
- lm-eval `num_fewshot`: `0`
- Generation: `max_gen_toks=16384`
- Serving: vLLM OpenAI-compatible chat server with `--generation-config vllm`
- OpenAI chat payload: `dataos_official_chat_payload=true`, so the DataOS
  lm-eval backend sends only `model`, `messages`, and `max_tokens` to
  `/v1/chat/completions`, matching the official evaluator. Do not inject
  lm-eval's default `temperature=0`, `seed`, or `stop` fields for HealthBench.
- Chat formatting: model-family templates only (`qwen3.jinja`, `smollm.jinja`,
  `gemma3.jinja`)
- Judge: OpenAI-compatible `qwen3.5-plus` through DashScope/Bailian
- Judge base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Judge request for non-GPT judges: `temperature=0`, `max_tokens=512`,
  JSON-object response format, and thinking disabled through provider-specific
  `extra_body`
- Judge retry policy: 5 retries with 2 second linear backoff
- Judge concurrency policy: at most 8 rubric workers per sample and at most 4
  concurrent judge requests
- Scorer: per-rubric criteria-met judge, weighted by HealthBench points and
  normalized by positive-point rubrics

This protocol keeps the official HealthBench dataset order, prompt shape,
rubric prompt, max generation length, and scoring rule. It differs from
PostTrainBench's official final evaluator in the judge model/API:
PostTrainBench uses `gpt-5-mini` with OpenAI API credentials and much higher
judge concurrency, while DataOS uses `qwen3.5-plus` through DashScope/Bailian.
Therefore HealthBench result tables must describe this as a fixed project judge
protocol, not as an exact official-score reproduction.

The task writes per-rubric evidence to `healthbench_judge_events.jsonl`, and the
protocol runner records the non-secret judge policy in `protocol.json`. API keys
must be supplied by environment variables and must not be committed or logged as
values.

Current HealthBench evidence:

- local report:
  `reports/posttrainbench_healthbench_fixed_protocol_alignment_2026-05-18.html`
- official direct diagnostic:
  `/home/hlv/healthbench_official_direct_diag_20260518`
- fixed official-payload paired sanity:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/healthbench_official_payload_sanity_20260518_1305`
- fixed official-payload four-Qwen limit-32 run root:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/healthbench_official_payload_qwen4_limit32_20260518_1310`
- accepted eight-model full run:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/healthbench_official_payload_full8_20260518_1408`
- old pre-fix diagnostic evidence, retained only as a rejected counterexample:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/healthbench_qwen35_qwen4_limit32_20260518_1150`

### ArenaHard-Writing DataOS Protocol

`posttrainbench_arena_hard_writing.yaml` is the DataOS/lm-eval ArenaHard-Writing
task. It is a working DataOS -> lm-eval -> vLLM -> OpenAI-compatible pairwise
judge pipeline, but not a strict numerical reproduction of upstream
PostTrainBench while the judge provider/model differs.

- Dataset: PostTrainBench local `arena-hard-v2.0/question.jsonl`
- Split/order: the official local JSONL shuffled with `random.Random(42)`
- Baseline answers: `Qwen3-1.7B` for `creative_writing`, and
  `o3-mini-2025-01-31` for `hard_prompt`, `coding`, and `math`
- Prompt: each sample is sent as a single user chat message
- lm-eval `num_fewshot`: `0`
- Generation: `max_gen_toks=16384`
- Serving: vLLM OpenAI-compatible chat server with `--generation-config vllm`
- OpenAI chat payload: `dataos_official_chat_payload=true`, so the DataOS
  backend sends only `model`, `messages`, and `max_tokens` to
  `/v1/chat/completions`, matching the official evaluator. Do not inject
  lm-eval's default `temperature=0`, `seed`, or `stop` fields for ArenaHard.
- Chat formatting: model-family templates only (`qwen3.jinja`, `smollm.jinja`,
  `gemma3.jinja`)
- Judge prompt: the official `arena-hard-v2.0.yaml` pairwise prompt template
  plus the official category-specific system prompt from `judge_utils.py`
- Judge games: two games per sample, baseline-vs-candidate and
  candidate-vs-baseline, then the official weighted `A>>B/A>B/A=B/B>A/B>>A`
  score map
- Judge: OpenAI-compatible `qwen3.5-plus` through DashScope/Bailian for the
  project protocol; official PostTrainBench uses `gpt-5-mini`
- Judge base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Judge request for non-GPT judges: `temperature=0`, `max_tokens=16000`, and
  thinking disabled through provider-specific `extra_body`
- Judge retry policy: 5 retries with 2 second linear backoff
- Judge concurrency policy: 1 concurrent judge request while HealthBench full
  runs are active, to avoid API-rate interference

This protocol keeps the official ArenaHard-Writing dataset order, prompt shape,
baseline policy, pairwise judge prompt, max generation length, and scoring rule.
It differs from PostTrainBench's official final evaluator in the judge
model/API, so tables must describe this as a fixed project judge protocol.

The task writes per-game evidence to `arena_judge_events.jsonl`, and the
protocol runner records the non-secret judge policy in `protocol.json`. API keys
must be supplied by environment variables and must not be committed or logged as
values.

Current ArenaHard-Writing evidence:

- local report:
  `reports/posttrainbench_arena_hard_writing_fixed_protocol_alignment_2026-05-18.html`
- fixed official-payload one-sample sanity:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/arena_official_payload_sanity_20260518_1535`
- accepted eight-model full run:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/arena_official_payload_full8_20260519_073202`
- safety-skip rerun for `SmolLM3-3B-Base`:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/arena_official_payload_full8_20260519_073202/SmolLM3-3B-Base/arenahardwriting`
- prior partial DataOS/lm-eval proxy judge run, retained only as pre-fix
  evidence:
  `/home/hlv/dataos_lmeval_matrix_runs_20260514/taskwise_non_aime_20260516_145516/arenahardwriting_bailian_qwen35_full`

## Maintenance Rules

1. Keep benchmark-specific logic in this directory or in a dedicated runner.
   The generic `lm_eval_native` backend should remain benchmark-agnostic.
2. Add one PostTrainBench benchmark at a time.
3. For each new benchmark, include the task YAML, helper functions, template
   requirements, and a short validation note.
4. Record reference model results before treating a protocol as aligned.
5. Preserve `protocol.json` artifacts from validation runs outside the package
   when large sample outputs are involved. Do not commit large run artifacts or
   model outputs to the repository.

## Known Limitations

The suite is now represented in lm-eval. All seven tasks have server-side smoke
tests and accepted eight-model fixed-protocol result artifacts. HumanEval uses
the robust scorer documented
above, BFCL has the documented `max_connections=1` caveat, GPQA Main has the
documented frozen-choice-order caveat, AIME is the DataOS fixed K32 protocol,
and HealthBench/Arena-Hard Writing are fixed DashScope `qwen3.5-plus`
judge-protocol results rather than strict official `gpt-5-mini` reproductions.
Arena-Hard Writing can skip content-safety judge failures only for the
documented DashScope `data_inspection_failed` class; authentication, quota,
billing, 429, and rate-limit failures remain fatal.
