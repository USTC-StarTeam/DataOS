# PostTrainBench chat templates

`dataos-eval` picks one of these from `../../protocol.yaml` (`model_families[*].eval_template`),
keyed on the model's own `config.json` `model_type`, and passes it to the runner as
`--chat-template`. It is the exact wrapper the scorer sends at eval time.

It used to be picked by substring of `f"{model_name} {model_path}"` — and `--model-name` is filled
by the experimenter, so an agent could change the eval protocol by choosing a label. The template
is also the ONLY thing that decides the wrapper now: the runner overrides whatever chat template
the model directory happens to carry, because that artifact is a copy of the base's, a lossy
rebuild by LLaMA-Factory, or absent, depending on the base model — never the declared one.

## Train ↔ eval pairing (get this wrong and the score is capped by protocol, not capability)

Whatever a run trains with must emit the same turn tokens this file does. The training side is
LLaMA-Factory's `template:` field, and the two namespaces do **not** line up by name:

| model family | eval template (here) | turn tokens | LF training template |
|---|---|---|---|
| Qwen3 | `qwen3.jinja` | `<|im_start|>` / `<|im_end|>` | `qwen3`, `qwen3_nothink` (LF built-in) |
| SmolLM3 | `smollm.jinja` | `<|im_start|>` / `<|im_end|>` | **`smollm3`, `smollm3_nothink`** — registered by dataos-train, NOT LF built-ins |
| Gemma3 | `gemma3.jinja`, `gemma3_tool_calling.jinja` | `<start_of_turn>` / `<end_of_turn>` | `gemma` (LF built-in) |

⚠️ **LF has no `smollm3` template** — not in 0.9.4, not in the 0.9.5 release, not on `main`
(checked 2026-08-11). An earlier version of this note said its `smollm` / `smollm2` were the
SmolLM1/2 Llama-style `<|eot_id|>` format; that was **wrong**. Their turn slots are ChatML and
byte-identical to `qwen3`'s. The two things that actually make them unusable for SmolLM3-3B-Base:

| | why it breaks |
|---|---|
| `replace_eos=False` | the base's `eos_token` is `<\|end_of_text\|>` (`<\|im_end\|>` is in the vocab, id 128012, but is not eos) — the saved `generation_config` then never stops generation |
| `tool_format="default"` | LF's default is ReAct prose (`Action:` / `Action Input:`); SmolLM3 wants `<tools>` XML + `<tool_call>{"name":..,"arguments":..}</tool_call>`, i.e. LF's `qwen` format |

LF exposes no per-run override for either, so `dataos_train.backends.llamafactory.ext.templates`
registers `smollm3` / `smollm3_nothink` with them fixed, and the LF backend routes those two
through a wrapper entry so the registration is in place; every other template keeps using
`llamafactory-cli` directly.

The registration also carries `default_system` — a hand translation of the header `smollm.jinja`
injects into EVERY request (see the note below). `tools/tests/train/test_train_eval_template_parity.py`
renders both sides on the same messages and fails if they drift; it caught one divergence already
(the `/no_think` branch prefills an empty `<think></think>`, which LF's own `qwen3_nothink` does
not model).

## Notes per template

**`smollm.jinja`** — SmolLM3's official template. Two things to know:

1. It **always** emits a system turn, even when the request has no system message: a
   `## Metadata` block (knowledge cutoff + *today's date* + reasoning mode) followed by
   `## Custom Instructions`. With no system message and `enable_thinking` defaulting to true,
   those instructions are SmolLM's persona plus a demand to answer as
   `<think> Thought </think> Solution`. Benches that parse a short target (`ANSWER: N`) are
   evaluated under that instruction whether or not the run trained for it. A system message
   containing `/system_override` replaces the whole block; `/think` / `/no_think` in a system
   message switches the reasoning mode. The eval harness sends no system message for
   gsm8k / humaneval / gpqamain / aime2025, so those always get the default preamble.
2. Upstream emitted the system turn's closing `<|im_end|>` **inside** the tools branch, so any
   prompt without tools left the system turn unterminated (`...persona<|im_start|>user`). Fixed
   here 2026-08-10 — before that no training template could match a non-tool bench's prompt.

That preamble went unmirrored on the training side until 2026-08-11: the LF template declared
`default_system=""`, so a training prompt was **55 bytes** against the scorer's **1360** — the
model was trained without the instruction it was then scored under, across all 7 benches of the
SmolLM3 batch. It is now hand-translated into the LF registration, both reasoning branches, and
the parity test holds the two together. The one field that cannot match is `Today Date`: the jinja
renders it when the scorer runs, dataos-train renders it when training starts. Same day →
identical; different day → one line differs.

Note also that only arenahardwriting, healthbench and humaneval strip `<think>` before parsing
(`_strip_thinking`, and only when the text *starts* with `<think>` and closes it). gsm8k,
gpqamain, aime2025 and bfcl do not — so a thought block that never closes inside `max_gen_toks`
takes the answer with it.
