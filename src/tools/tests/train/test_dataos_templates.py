"""The chat templates LF does not ship, and the routing that registers them.

Training must emit the same turn tokens the scorer sends. LF 0.9.4 has no SmolLM3 template and
its same-named `smollm` is the SmolLM1/2 `<|eot_id|>` format, so picking by name breaks the run.
See ext/templates.py.
"""

from __future__ import annotations

import json

import pytest

from dataos_train.backends.llamafactory import LlamaFactoryBackend
from dataos_train.backends.llamafactory.ext.templates import (
    DATAOS_TEMPLATES,
    resolve_template,
)


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
    assert "ext.lf_entry train" in b._train_entry(_Spec(template), template)
    assert "ext.lf_entry export" in b._export_entry(template)


@pytest.mark.parametrize("template", ["qwen3", "gemma", "smollm"])
def test_lf_owned_templates_keep_lf_cli(template):
    """Only what needs the wrapper goes through it — everything else is untouched."""
    b = _backend()
    assert b._train_entry(_Spec(template), template) == "llamafactory-cli train"
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


# --- deriving the template from the model itself ---------------------------------------------
# The default used to be a hardcoded `template: str = "qwen3"` that no plan ever overrode
# (68/68 in the smol3 batch). Right for Qwen, accidentally right for SmolLM3, silently wrong for
# Gemma — and a wrong chat template caps every score with nothing in any log.


def _model_dir(tmp_path, model_type: str):
    d = tmp_path / model_type
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
    return d


@pytest.mark.parametrize(
    ("model_type", "expected"), [("qwen3", "qwen3"), ("smollm3", "smollm3"), ("gemma3", "gemma3")]
)
def test_template_comes_from_the_models_own_config(tmp_path, model_type, expected):
    assert resolve_template(_model_dir(tmp_path, model_type)) == expected


def test_a_continuation_node_resolves_from_the_saved_parent(tmp_path):
    """A continuation's model path is the PARENT's final_model dir and carries no family name
    — which is why this keys on model_type and not on a path substring."""
    parent = tmp_path / "round-0003" / "some-node" / "final_model"
    parent.mkdir(parents=True)
    (parent / "config.json").write_text(json.dumps({"model_type": "smollm3"}), encoding="utf-8")
    assert "smollm" not in str(parent).lower()  # the path gives nothing away
    assert resolve_template(parent) == "smollm3"


def test_unknown_family_raises_instead_of_falling_back(tmp_path):
    with pytest.raises(ValueError, match="no chat template known for model_type"):
        resolve_template(_model_dir(tmp_path, "llama"))


def test_unreadable_model_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="unreadable"):
        resolve_template(tmp_path / "nope")


def test_build_derives_the_template_when_the_plan_omits_it(tmp_path):
    from dataos_train.spec import DatasetRef, TrainSpec

    from dataos_train.spec import SFT

    model = _model_dir(tmp_path, "smollm3")
    data = tmp_path / "data"
    data.mkdir()
    spec = TrainSpec(
        name="n", base_model=str(model), run_dir=str(tmp_path / "run"),
        dataset=DatasetRef(path=str(data)), method=SFT(finetuning_type="lora"),
    )
    assert spec.method.template is None  # the plan said nothing
    job = LlamaFactoryBackend().build(spec)
    assert "template: smollm3" in job.assets["train_config.yaml"]
    # and the routing follows the RESOLVED value, not the un-copied spec
    assert "ext.lf_entry train" in job.assets["run_train.sh"]
