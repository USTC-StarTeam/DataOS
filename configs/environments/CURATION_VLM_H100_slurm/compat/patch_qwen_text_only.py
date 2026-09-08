"""Auditable compatibility patch for the pinned Qwen mixed-modality training loader."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ORIGINAL_SHA256 = "6aabb3317e993a4802ab7db8971197263bbefeed438f1b3afe6309a6334fca9d"
TARGET = Path("src/curation_train/train_qwen2vl.py")
IMAGE_ORIGINAL = """        if not isinstance(images, list):
            images = [images]
"""
IMAGE_REPLACEMENT = """        images = [] if images is None else (images if isinstance(images, list) else [images])
        images = [image for image in images if image is not None]
"""
ARGS_ORIGINAL = """    script_args, training_args = parser.parse_args_into_dataclasses(return_remaining_strings=False)
"""
ARGS_REPLACEMENT = ARGS_ORIGINAL + """
    # Text-only microbatches do not execute the vision tower on every DDP rank.
    training_args.ddp_find_unused_parameters = True
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            **(training_args.gradient_checkpointing_kwargs or {}), "use_reentrant": False,
        }
"""


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def render(original: bytes) -> bytes:
    if sha256(original) != ORIGINAL_SHA256:
        raise ValueError("unexpected Qwen trainer revision; refusing compatibility patch")
    text = original.decode("utf-8")
    for old, new in ((IMAGE_ORIGINAL, IMAGE_REPLACEMENT), (ARGS_ORIGINAL, ARGS_REPLACEMENT)):
        if text.count(old) != 1:
            raise ValueError("expected exactly one Qwen compatibility patch site")
        text = text.replace(old, new)
    compile(text, str(TARGET), "exec")
    return text.encode("utf-8")


def patch(project: Path) -> Path:
    target = project / TARGET
    backup = target.with_suffix(".py.pre-text-only")
    original = backup.read_bytes() if backup.exists() else target.read_bytes()
    patched = render(original)
    actual = target.read_bytes()
    if actual not in (original, patched):
        raise ValueError("Qwen trainer has unrelated modifications; refusing to overwrite")
    if not backup.exists():
        with backup.open("xb") as stream:
            stream.write(original)
    if actual != patched:
        target.write_bytes(patched)
    print(f"Qwen mixed-modality compatibility ready: {sha256(patched)}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    patch(parser.parse_args().project)
