"""Read-only offline checks for the downloaded checkpoint and its dedicated runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path


def verify_weights(model: Path) -> None:
    revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    assert len(shards) == 18, f"unexpected shard count: {len(shards)}"
    for name in shards:
        metadata = model / ".cache/huggingface/download" / f"{name}.metadata"
        commit, digest, *_ = metadata.read_text().splitlines()
        assert commit == revision, (name, commit)
        with (model / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == digest, f"checksum mismatch: {name}"
        print(f"sha256 OK: {name}", flush=True)


def verify_runtime(model: Path, repo: Path) -> None:
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from llamafactory.data.template import TEMPLATES

    config = AutoConfig.from_pretrained(model, local_files_only=True)
    assert config.model_type == "qwen3_5"
    AutoProcessor.from_pretrained(model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    frozen = repo / "configs/tasks/posttrainbench/eval/lmeval/templates/qwen3_8.jinja"
    assert frozen.read_text().rstrip() == (model / "chat_template.jinja").read_text().rstrip()
    tokenizer.chat_template = frozen.read_text()
    template = copy.deepcopy(TEMPLATES["qwen3_8"])
    template.fix_special_tokens(tokenizer)
    tool = {
        "name": "echo", "description": "Return the supplied text.",
        "parameters": {
            "type": "object", "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    }
    for tools in (None, [tool]):
        question = {"role": "user", "content": "Return hello."}
        answer = {"role": "assistant", "content": "<think>\nCheck the request.\n</think>\n\nhello"}
        prompt_ids, response_ids = template.encode_oneturn(
            tokenizer, [question, answer], tools=json.dumps(tools) if tools else None,
        )
        expected = tokenizer.apply_chat_template(
            [question], tools=[{"type": "function", "function": tool}] if tools else None,
            tokenize=False, add_generation_prompt=True,
        )
        actual = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        training_text = tokenizer.decode(prompt_ids + response_ids, skip_special_tokens=False)
        # LF supervises the opening <think> token; serving prefills it. Compare the whole
        # training prefix so this loss-mask boundary is not mistaken for a framing mismatch.
        assert expected.startswith(actual) and training_text.startswith(expected), (
            f"train/eval prompt mismatch: {actual!r} != {expected!r}"
        )
    print("Qwen3.8 tokenizer, processor, and training/eval template parity OK", flush=True)
    for package in ("torch", "transformers", "llamafactory", "vllm", "lm-eval"):
        print(f"{package}={importlib.metadata.version(package)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    verify_weights(args.model)
    verify_runtime(args.model, args.repo)


if __name__ == "__main__":
    main()
