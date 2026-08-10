# PostTrainBench chat templates

`run_lmeval_eval.py::resolve_chat_template` picks one of these by a substring of the model
name/path (`qwen` / `smollm` / `gemma`) and hands it to vLLM as `--chat-template`. It is the exact
wrapper the scorer sends at eval time.

## Train ↔ eval pairing (get this wrong and the score is capped by protocol, not capability)

Whatever a run trains with must emit the same turn tokens this file does. The training side is
LLaMA-Factory's `template:` field, and the two namespaces do **not** line up by name:

| model family | eval template (here) | turn tokens | LF training template |
|---|---|---|---|
| Qwen3 | `qwen3.jinja` | `<|im_start|>` / `<|im_end|>` | `qwen3`, `qwen3_nothink` (LF built-in) |
| SmolLM3 | `smollm.jinja` | `<|im_start|>` / `<|im_end|>` | **`smollm3`, `smollm3_nothink`** — registered by dataos-train, NOT LF built-ins |
| Gemma3 | `gemma3.jinja`, `gemma3_tool_calling.jinja` | `<start_of_turn>` / `<end_of_turn>` | `gemma` (LF built-in) |

⚠️ **LF's built-in `smollm` / `smollm2` are the SmolLM1/2 Llama-style format**
(`stop_words=["<|eot_id|>", "<|eom_id|>"]`) — they do not match `smollm.jinja` and must not be
used for SmolLM3. `dataos_train.backends.llamafactory.ext.templates` registers `smollm3` /
`smollm3_nothink` (ChatML, `stop_words=["<|im_end|>"]`, `replace_eos=True`) and the LF backend
routes those two through a wrapper entry so the registration is in place; every other template
keeps using `llamafactory-cli` directly.

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

Because the date is part of the preamble, no static `default_system` can reproduce it; mirroring
the preamble in a recipe's `system` field is a method choice, not a template constant.
