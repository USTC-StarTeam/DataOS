"""Chat templates LLaMA-Factory does not ship, registered by dataos-train.

WHY THIS EXISTS
---------------
Training and evaluation must render the SAME chat format, or the model learns turn tokens the
scorer never sends and every score is capped by protocol rather than capability. The task's eval
side picks a jinja template by model family (``configs/tasks/posttrainbench/eval/lmeval/
templates/``), so for each family LF has to offer a training template that emits the same turn
tokens. It does for Qwen (``qwen3``) and Gemma. It does NOT for SmolLM3:

LF has no ``smollm3`` template — not in 0.9.4, not in 0.9.5, not on ``main`` (checked
2026-08-11; upstream simply has not added SmolLM3). Its similarly-named ``smollm`` / ``smollm2``
are ChatML and their turn slots are byte-identical to ``qwen3``'s, so the reason they are unusable
is NOT the turn framing (an earlier version of this note claimed they were the Llama-style
``<|eot_id|>`` format — that was wrong). The two real blockers, both measured:

* ``replace_eos=False`` — SmolLM3-3B-Base's ``eos_token`` is ``<|end_of_text|>``, while
  ``<|im_end|>`` is merely present in the vocab (id 128012). Without the swap the saved
  ``generation_config.eos_token_id`` still points at ``<|end_of_text|>``, so nothing stops
  generation when the model emits ``<|im_end|>``.
* ``tool_format="default"`` — LF's default tool format is ReAct prose
  (``Action: ... / Action Input: ...``), while SmolLM3 asks for ``<tools>`` XML plus
  ``<tool_call>{"name":..,"arguments":..}</tool_call>``, which is LF's ``qwen`` format. Training
  bfcl under the default format teaches a call syntax the scorer cannot parse at all.

LF exposes no per-run override for either, so registering our own IS the mechanism for setting
them. ``smollm3`` / ``smollm3_nothink`` therefore mirror LF's ``qwen3`` / ``qwen3_nothink`` pair
(same slots, same ``ReasoningTemplate`` / ``Template`` split) with those two blockers fixed.

WHY ``default_system`` IS NOT EMPTY
-----------------------------------
``smollm.jinja`` ALWAYS emits a system turn, even when the request carries no system message, and
with ``enable_thinking`` defaulting to true that turn is a ``## Metadata`` block plus a long
"answer as ``<think> Thought </think> Solution``" instruction. The eval harness sends no system
message for gsm8k / humaneval / gpqamain / aime2025, so every one of those prompts carries it.
Leaving ``default_system`` empty made the training prompt 55 bytes against the scorer's 1360 —
the model was trained without an instruction it was then evaluated under. The constants below are
a hand translation of both branches of that header; ``tests/test_train_eval_template_parity.py``
renders the jinja and this template on the same messages and fails if they drift apart.

The one field that cannot match is ``Today Date``: the jinja renders it at eval time, so this side
renders it at registration time. Same day → identical; different day → one line differs.

(Registration happens in-process, so it is lost across LF's torchrun re-exec — the entry that
calls :func:`register_all` asserts single-process before relying on it.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

#: Template names this module adds to LF's registry. The LF backend routes training/export
#: through ``ext.lf_entry`` instead of ``llamafactory-cli`` when a spec asks for one of these.
DATAOS_TEMPLATES: tuple[str, ...] = ("smollm3", "smollm3_nothink")

#: HF ``model_type`` -> LLaMA-Factory template. THE DECLARATION lives in the task config
#: (``configs/tasks/<task>/eval/protocol.yaml`` -> ``model_families[*].train_template``), which
#: declares both halves of the pair on one line; this dict mirrors its train half because this
#: package runs in a venv that has no access to that file at import time.
#: ``tests/agent/test_eval_protocol_config.py`` fails if the two drift.
#: Both halves must render the same turn tokens or the score is capped by protocol instead of
#: capability. Keyed on ``model_type`` rather than a path substring
#: because a continuation node's model path is the PARENT's ``final_model`` dir and carries no
#: family name, while a saved model always keeps its ``model_type``.
MODEL_TYPE_TEMPLATES: dict[str, str] = {
    "qwen3": "qwen3",
    "smollm3": "smollm3",  # registered below; LF's own `smollm` is the SmolLM1/2 format
    "gemma3": "gemma",
}


def model_family(model_path: str | Path) -> str:
    """The model's own HF ``model_type`` — the key everything about a family hangs off.

    Read from the model rather than taken from a caller's string: an agent-supplied name can
    be wrong (or absent, for a continuation node whose path is the parent's ``final_model``),
    and both things it selects — the chat template below, and the training env when the
    environment declares family-scoped env domains — fail SILENTLY or expensively when they
    are wrong.
    """
    cfg = Path(model_path) / "config.json"
    try:
        model_type = json.loads(cfg.read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"cannot derive the model family: {cfg} is unreadable ({exc}). Run build where "
            "the model path is visible."
        ) from exc
    if not model_type:
        raise ValueError(f"{cfg} declares no model_type; cannot derive the model family.")
    return str(model_type)


def resolve_template(model_path: str | Path, family: str | None = None) -> str:
    """The LF template for the model at ``model_path``, from its own ``config.json``.

    Raises rather than guessing. A wrong template is a SILENT failure — the model learns turn
    tokens the scorer never sends and every score is capped, with nothing in any log. The
    previous behaviour was a hardcoded ``template: str = "qwen3"`` default in ``spec.py``,
    which no plan ever overrode (68/68 in the 2026-08-11 smol3 batch): right for Qwen,
    accidentally right for SmolLM3 (also ChatML), and silently wrong for Gemma.

    ``family`` short-circuits the ``config.json`` read for callers that already resolved it.
    """
    model_type = family or model_family(model_path)
    template = MODEL_TYPE_TEMPLATES.get(str(model_type))
    if not template:
        raise ValueError(
            f"no chat template known for model_type {model_type!r} ({model_path}). Add it to "
            "MODEL_TYPE_TEMPLATES together with the matching eval-side jinja, or set "
            "method.template explicitly — do NOT let it fall back to another family's format."
        )
    return template


#: Hand translation of ``smollm.jinja``'s system header. The jinja emits it for EVERY request,
#: so the training side has to emit it too or the model never sees the instruction it is scored
#: under. Kept as three pieces because the two reasoning modes differ only in the mode line and
#: the persona; the parity test is what proves the reassembly still matches the jinja.
_SMOLLM3_SYSTEM = (
    "## Metadata\n\n"
    "Knowledge Cutoff Date: June 2025\n"
    "Today Date: {today}\n"
    "Reasoning Mode: {mode}\n\n"
    "## Custom Instructions\n\n"
    "{persona}\n\n"
)
_SMOLLM3_PERSONA = "You are a helpful AI assistant named SmolLM, trained by Hugging Face."
_SMOLLM3_PERSONA_THINK = _SMOLLM3_PERSONA + (
    " Your role as an assistant involves thoroughly exploring questions through a systematic "
    "thinking process before providing the final precise and accurate solutions. This requires "
    "engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, "
    "reflection, backtracking, and iteration to develop well-considered thinking process. Please "
    "structure your response into two main sections: Thought and Solution using the specified "
    "format: <think> Thought section </think> Solution section. In the Thought section, detail "
    "your reasoning process in steps. Each step should include detailed considerations such as "
    "analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the "
    "accuracy of the current steps, refining any errors, and revisiting previous steps. In the "
    "Solution section, based on various attempts, explorations, and reflections from the Thought "
    "section, systematically present the final solution that you deem correct. The Solution "
    "section should be logical, accurate, and concise and detail necessary steps needed to reach "
    "the conclusion."
)

_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _today() -> str:
    """``strftime("%d %B %Y")`` without the locale dependence.

    The eval side renders this field through the template's ``strftime_now``, i.e. the C locale on
    the cluster. Spelling the month out here means the training side cannot desync just because a
    job inherited a different ``LC_TIME``.
    """
    t = time.localtime()
    return f"{t.tm_mday:02d} {_MONTHS[t.tm_mon - 1]} {t.tm_year}"


def smollm3_default_system(reasoning_mode: str = "/think") -> str:
    """The system turn ``smollm.jinja`` injects when the request carries no system message."""
    persona = _SMOLLM3_PERSONA_THINK if reasoning_mode == "/think" else _SMOLLM3_PERSONA
    return _SMOLLM3_SYSTEM.format(today=_today(), mode=reasoning_mode, persona=persona)


#: ``smollm.jinja``'s ``/no_think`` branch prefills an empty thought block after the generation
#: prompt, so a run that evaluates with ``enable_thinking=False`` must train with it too. LF's own
#: ``qwen3_nothink`` does NOT model this (it reuses the thinking variant's slots) — the jinja is
#: the protocol here, not LF's convention, and the parity test is what says so.
_NOTHINK_PREFILL = "<think>\n\n</think>\n"


def chatml_slots(reasoning_mode: str = "/think") -> dict[str, list[str]]:
    """SmolLM3's ChatML slots for one reasoning mode — the hand translation of ``smollm.jinja``."""
    prompt = "<|im_start|>assistant\n" + (_NOTHINK_PREFILL if reasoning_mode == "/no_think" else "")
    return {
        "format_user": [f"<|im_start|>user\n{{{{content}}}}<|im_end|>\n{prompt}"],
        "format_assistant": ["{{content}}<|im_end|>\n"],
        "format_system": ["<|im_start|>system\n{{content}}<|im_end|>\n"],
        "format_observation": [
            f"<|im_start|>user\n<tool_response>\n{{{{content}}}}\n</tool_response><|im_end|>\n{prompt}"
        ],
    }


def register_all() -> tuple[str, ...]:
    """Register the missing templates in LF's registry. Returns the names registered.

    Imports llamafactory lazily: this package is also installed in envs without it.
    """
    from llamafactory.data.formatter import FunctionFormatter, StringFormatter, ToolFormatter
    from llamafactory.data.template import TEMPLATES, ReasoningTemplate, Template, register_template

    for name, template_class, mode in (
        ("smollm3", ReasoningTemplate, "/think"),
        ("smollm3_nothink", Template, "/no_think"),
    ):
        if name in TEMPLATES:  # an LF upgrade may ship its own — never shadow it
            continue
        slots = chatml_slots(mode)
        register_template(
            name=name,
            # the header smollm.jinja injects for every request; empty here meant the model was
            # trained without the instruction it is scored under (55 bytes vs the scorer's 1360)
            default_system=smollm3_default_system(mode),
            format_user=StringFormatter(slots=slots["format_user"]),
            format_assistant=StringFormatter(slots=slots["format_assistant"]),
            format_system=StringFormatter(slots=slots["format_system"]),
            # SmolLM3 asks for calls as {"name":..,"arguments":..} inside <tool_call></tool_call>
            # (smollm.jinja's Tools section) — byte-identical to LF's "qwen" tool format.
            format_function=FunctionFormatter(slots=slots["format_assistant"], tool_format="qwen"),
            format_observation=StringFormatter(slots=slots["format_observation"]),
            format_tools=ToolFormatter(tool_format="qwen"),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            template_class=template_class,
        )
    return DATAOS_TEMPLATES
