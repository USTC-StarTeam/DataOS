from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[2] / (
    "configs/environments/CURATION_VLM_H100_slurm/compat/patch_qwen_text_only.py"
)


def _module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("qwen_compat", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_null_image_forms_preserve_text_and_real_images():
    module = _module()
    namespace = {}
    exec("def normalize(images):\n" + textwrap.indent(
        textwrap.dedent(module.IMAGE_REPLACEMENT), "    "
    ) + "    return images\n", namespace)
    normalize = namespace["normalize"]
    for value in (None, [], [None], [None, None]):
        assert normalize(value) == []
    image = object()
    assert normalize(image) == [image]
    assert normalize([None, image, None]) == [image]
    assert normalize("image.png") == ["image.png"]


def _fixture(module, monkeypatch):
    source = ("class Dataset:\n    def load(self, images):\n" + module.IMAGE_ORIGINAL
              + "\ndef train():\n" + module.ARGS_ORIGINAL).encode()
    monkeypatch.setattr(module, "ORIGINAL_SHA256", hashlib.sha256(source).hexdigest())
    return source


def test_patch_is_pinned_idempotent_and_preserves_backup(tmp_path, monkeypatch):
    module = _module()
    source = _fixture(module, monkeypatch)
    target = tmp_path / module.TARGET
    target.parent.mkdir(parents=True)
    target.write_bytes(source)
    module.patch(tmp_path)
    actual = target.read_bytes()
    module.patch(tmp_path)
    assert target.read_bytes() == actual == module.render(source)
    assert target.with_suffix(".py.pre-text-only").read_bytes() == source
    target.write_bytes(actual + b"\n# unrelated user edit\n")
    with pytest.raises(ValueError, match="unrelated modifications"):
        module.patch(tmp_path)


def test_patch_refuses_unknown_trainer(tmp_path):
    module = _module()
    with pytest.raises(ValueError, match="unexpected Qwen trainer"):
        module.render(b"unknown trainer")


@pytest.mark.parametrize("checkpointing", [False, True])
def test_compatibility_only_changes_distributed_runtime_flags(monkeypatch, checkpointing):
    module = _module()
    source = _fixture(module, monkeypatch)
    args = SimpleNamespace(
        gradient_checkpointing=checkpointing,
        gradient_checkpointing_kwargs={"preserve_rng_state": True},
        ddp_find_unused_parameters=None,
        learning_rate=1e-5, num_train_epochs=1, gradient_accumulation_steps=8,
    )
    parser = SimpleNamespace(parse_args_into_dataclasses=lambda **kw: (object(), args))
    namespace = {"parser": parser}
    exec(module.render(source), namespace)
    namespace["train"]()
    assert args.ddp_find_unused_parameters is True
    assert args.gradient_checkpointing_kwargs["preserve_rng_state"] is True
    assert args.gradient_checkpointing_kwargs.get("use_reentrant") == (
        False if checkpointing else None
    )
    assert (args.learning_rate, args.num_train_epochs, args.gradient_accumulation_steps) == (1e-5, 1, 8)
