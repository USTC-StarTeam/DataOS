"""The chat templates LF does not ship, and the routing that registers them.

Training must emit the same turn tokens the scorer sends. LF 0.9.4 has no SmolLM3 template and
its same-named `smollm` is the SmolLM1/2 `<|eot_id|>` format, so picking by name breaks the run.
See ext/templates.py.
"""

from __future__ import annotations

import pytest

from dataos_train.backends.llamafactory import LlamaFactoryBackend
from dataos_train.backends.llamafactory.ext.templates import DATAOS_TEMPLATES


class _Method:
    def __init__(self, template: str) -> None:
        self.template = template
        self.finetuning_type = "full"
        self.kind = "sft"


class _Spec:
    def __init__(self, template: str) -> None:
        self.method = _Method(template)
        self.online_ops = None


def _backend() -> LlamaFactoryBackend:
    return LlamaFactoryBackend.__new__(LlamaFactoryBackend)


@pytest.mark.parametrize("template", DATAOS_TEMPLATES)
def test_our_templates_route_through_the_registering_entry(template):
    b = _backend()
    assert "ext.lf_entry train" in b._train_entry(_Spec(template))
    assert "ext.lf_entry export" in b._export_entry(template)


@pytest.mark.parametrize("template", ["qwen3", "gemma", "smollm"])
def test_lf_owned_templates_keep_lf_cli(template):
    """Only what needs the wrapper goes through it — everything else is untouched."""
    b = _backend()
    assert b._train_entry(_Spec(template)) == "llamafactory-cli train"
    assert b._export_entry(template) == "llamafactory-cli export"


def test_registration_produces_chatml_with_the_smollm3_stop_token():
    """Cluster-only (needs llamafactory): the registered template must be ChatML, not
    SmolLM1/2's <|eot_id|> — that mismatch is what caps every score."""
    pytest.importorskip("llamafactory")
    from llamafactory.data.template import TEMPLATES

    from dataos_train.backends.llamafactory.ext.templates import register_all

    register_all()
    for name in DATAOS_TEMPLATES:
        t = TEMPLATES[name]
        assert t.stop_words == ["<|im_end|>"], name
        assert t.replace_eos is True, name  # -Base eos is <|end_of_text|>; the turn token is taught
        assert "<|im_start|>user" in "".join(map(str, t.format_user.slots)), name
        assert "<|eot_id|>" not in "".join(map(str, t.format_user.slots)), name
