"""The gate: what training renders must equal what the scorer sends.

Training renders through LLaMA-Factory's SLOT templates; evaluation renders through the family's
official jinja. They are two representations of one format, and the only thing that had ever kept
them in agreement was a person reading both files. That failed silently and expensively:

* ``smollm.jinja`` emits a system turn for EVERY request (a ``## Metadata`` block plus, with
  ``enable_thinking`` defaulting to true, a long "answer as ``<think>…</think>`` " instruction),
  while the LF side declared ``default_system=""``. Training prompt 55 bytes, eval prompt 1360.
  Nothing reported it; it surfaced only when someone rendered both by hand.

Rendering the jinja needs no tokenizer and no transformers — only a no-op ``{% generation %}``
tag, which is a marker for deriving assistant-token masks and contributes nothing to the output.
The LF slots for the family we hand-registered are plain constants in this package, so THAT family
is gated everywhere. LF's built-in families (qwen3, gemma) need llamafactory and are skipped where
it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, nodes
from jinja2.ext import Extension

REPO = Path(__file__).resolve().parents[4]
TASK = REPO / "configs" / "tasks" / "posttrainbench"
TEMPLATES = TASK / "eval" / "lmeval" / "templates"

#: The date is the one field that cannot agree: the jinja renders it when the scorer runs, this
#: package renders it when training starts. Same day → identical, so parity is checked with the
#: field pinned rather than pretending the mismatch does not exist.
_FIXED_DATE = "01 January 2000"


class _GenerationTag(Extension):
    """``{% generation %}`` marks the assistant span for ``return_assistant_tokens_mask``. It is
    transparent to rendering, so the gate treats it as a pass-through instead of pulling in
    transformers just to compile one tag."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        call = self.call_method("_passthrough", [])
        return nodes.CallBlock(call, [], [], body).set_lineno(lineno)

    def _passthrough(self, caller):
        return caller()


def render_protocol(template: str, **kwargs) -> str:
    """The prompt the scorer actually sends for a single user turn."""
    env = Environment(extensions=[_GenerationTag], trim_blocks=True, lstrip_blocks=True)
    env.globals["strftime_now"] = lambda _fmt: _FIXED_DATE
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(AssertionError(msg))
    return env.from_string((TEMPLATES / template).read_text(encoding="utf-8")).render(
        messages=[{"role": "user", "content": "U"}],
        add_generation_prompt=True,
        bos_token="",
        **kwargs,
    )


def render_slots(template, content: str = "U") -> str:
    """The prompt LF builds from its slots for the same turn: prefix + system + user.

    Mirrors ``Template._encode`` for the single-turn, no-system-column case — which is every row
    dataos-train produces, because the SFT ``dataset_info.json`` declares no ``columns`` map and
    LF therefore never sees a per-row ``system`` field.
    """
    out = ""
    for slot in getattr(template.format_prefix, "slots", []):
        if isinstance(slot, str):  # a set/dict slot is a tokenizer special (e.g. bos), not text
            out += slot
    if template.default_system:
        out += template.format_system.apply(content=template.default_system)[0]
    return out + template.format_user.apply(content=content)[0]


def _protocol() -> dict:
    return yaml.safe_load((TASK / "eval" / "protocol.yaml").read_text(encoding="utf-8"))


def _lf_template(name: str):
    """LF's registry entry for ``name``, or a skip when llamafactory is not installed here."""
    pytest.importorskip("llamafactory", reason="LF built-in slots need llamafactory")
    from llamafactory.data.template import TEMPLATES as LF_TEMPLATES

    from dataos_train.backends.llamafactory.ext.templates import register_all

    register_all()
    return LF_TEMPLATES[name]


@pytest.mark.parametrize("family", sorted(_protocol()["model_families"]))
def test_training_prompt_equals_the_scorer_prompt(family: str) -> None:
    spec = _protocol()["model_families"][family]
    template = _lf_template(spec["train_template"])
    expected = render_protocol(spec["eval_template"])
    actual = render_slots(template).replace(_today_of(template), _FIXED_DATE)
    assert actual == expected, (
        f"{family}: the training prompt and the scorer's prompt disagree.\n"
        f"  train ({len(actual)}B): {actual!r}\n"
        f"  eval  ({len(expected)}B): {expected!r}\n"
        f"Fix the LF template ({spec['train_template']}) — the jinja is the protocol and does "
        "not move."
    )


def _today_of(template) -> str:
    """The date this process baked into ``default_system``, so it can be pinned for comparison."""
    from dataos_train.backends.llamafactory.ext.templates import _today

    return _today() if template.default_system else _FIXED_DATE


def test_smollm3_header_is_gated_without_llamafactory() -> None:
    """The family we hand-translated is checked even where LF is absent — its slots and its
    system header are constants in this package, so nothing about this needs a training backend."""
    from dataos_train.backends.llamafactory.ext.templates import (
        _today,
        chatml_slots,
        smollm3_default_system,
    )

    for mode, kwargs in (("/think", {}), ("/no_think", {"enable_thinking": False})):
        slots = chatml_slots(mode)
        system = smollm3_default_system(mode).replace(_today(), _FIXED_DATE)
        actual = slots["format_system"][0].replace("{{content}}", system) + slots["format_user"][
            0
        ].replace("{{content}}", "U")
        assert actual == render_protocol("smollm.jinja", **kwargs), (
            f"smollm3 ({mode}): the hand translation of smollm.jinja has drifted"
        )
