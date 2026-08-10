"""Chat templates LLaMA-Factory does not ship, registered by dataos-train.

WHY THIS EXISTS
---------------
Training and evaluation must render the SAME chat format, or the model learns turn tokens the
scorer never sends and every score is capped by protocol rather than capability. The task's eval
side picks a jinja template by model family (``configs/tasks/posttrainbench/eval/lmeval/
templates/``), so for each family LF has to offer a training template that emits the same turn
tokens. It does for Qwen (``qwen3``) and Gemma. It does NOT for SmolLM3:

* LF 0.9.4 ships ``smollm`` / ``smollm2`` — the SmolLM1/2 Llama-style format
  (``stop_words=["<|eot_id|>", "<|eom_id|>"]``).
* SmolLM3's official template (``smollm.jinja``) is ChatML: ``<|im_start|>`` / ``<|im_end|>``.

So the template whose NAME matches the model is the one that breaks it, while the correct choice
(``qwen3``) is named for a different family — a trap the Qwen batch never had to walk past. These
registrations remove the asymmetry: ``smollm3`` / ``smollm3_nothink`` mirror LF's own ``qwen3`` /
``qwen3_nothink`` pair, with SmolLM3's stop token.

``default_system`` is deliberately EMPTY. ``smollm.jinja`` injects a ``## Metadata`` block that
contains TODAY'S DATE, so no static constant can match it; whether to mirror that preamble in the
data's ``system`` field is a recipe decision, not a template constant.

``replace_eos=True`` matters for the -Base checkpoint: its ``eos_token`` is ``<|end_of_text|>``
while the instruct variant terminates turns with ``<|im_end|>``, so the turn token has to be
taught. (Registration happens in-process, so it is lost across LF's torchrun re-exec — the entry
that calls :func:`register_all` asserts single-process before relying on it.)
"""

from __future__ import annotations

#: Template names this module adds to LF's registry. The LF backend routes training/export
#: through ``ext.lf_entry`` instead of ``llamafactory-cli`` when a spec asks for one of these.
DATAOS_TEMPLATES: tuple[str, ...] = ("smollm3", "smollm3_nothink")

_CHATML = {
    "format_user": ["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"],
    "format_assistant": ["{{content}}<|im_end|>\n"],
    "format_system": ["<|im_start|>system\n{{content}}<|im_end|>\n"],
    "format_observation": [
        "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response>"
        "<|im_end|>\n<|im_start|>assistant\n"
    ],
}


def register_all() -> tuple[str, ...]:
    """Register the missing templates in LF's registry. Returns the names registered.

    Imports llamafactory lazily: this package is also installed in envs without it.
    """
    from llamafactory.data.formatter import FunctionFormatter, StringFormatter, ToolFormatter
    from llamafactory.data.template import TEMPLATES, ReasoningTemplate, Template, register_template

    for name, template_class in (("smollm3", ReasoningTemplate), ("smollm3_nothink", Template)):
        if name in TEMPLATES:  # an LF upgrade may ship its own — never shadow it
            continue
        register_template(
            name=name,
            format_user=StringFormatter(slots=_CHATML["format_user"]),
            format_assistant=StringFormatter(slots=_CHATML["format_assistant"]),
            format_system=StringFormatter(slots=_CHATML["format_system"]),
            # SmolLM3 asks for calls as {"name":..,"arguments":..} inside <tool_call></tool_call>
            # (smollm.jinja's Tools section) — byte-identical to LF's "qwen" tool format.
            format_function=FunctionFormatter(slots=_CHATML["format_assistant"], tool_format="qwen"),
            format_observation=StringFormatter(slots=_CHATML["format_observation"]),
            format_tools=ToolFormatter(tool_format="qwen"),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            template_class=template_class,
        )
    return DATAOS_TEMPLATES
