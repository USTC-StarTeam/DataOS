from __future__ import annotations

# ruff: noqa: E402

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import lm_eval

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataos.core.artifact import ModelArtifact
from dataos.core.sandbox import Sandbox
from dataos.eval.adapters.transformers import _render_chat_prompt
from dataos.eval.backend.native import _resolve_model_args
from dataos.eval.backend.runner import LmEvalRunOutput
from dataos.eval.config import EvalConfig
from dataos.eval.runner import _eval_output_to_artifact, run_eval_stage
from dataos.eval.types import AdapterCallContext, EvalAdapterConfig
from dataos.eval_tasks.posttrainbench import utils as posttrainbench_utils


def main() -> None:
    _check_native_stage_dispatch()
    _check_legacy_artifact_and_helpers()
    _check_posttrainbench_gsm8k_helpers()
    print("eval smoke check ok")


def _check_native_stage_dispatch() -> None:
    captured: dict[str, Any] = {}
    original_simple_evaluate = lm_eval.simple_evaluate

    def fake_simple_evaluate(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return {
            "results": {
                "gsm8k": {
                    "exact_match,flexible-extract": 0.5,
                    "alias": "gsm8k",
                }
            },
            "samples": {"gsm8k": [{"doc_id": 0, "prediction": "42"}]},
            "configs": {"gsm8k": {"task": "gsm8k"}},
            "versions": {"gsm8k": 1},
        }

    lm_eval.simple_evaluate = fake_simple_evaluate
    try:
        with tempfile.TemporaryDirectory(prefix="dataos-eval-native-") as tmpdir:
            root = Path(tmpdir)
            sandbox = Sandbox(root / "sandbox")
            sandbox.prepare()
            model_dir = root / "model"
            model_dir.mkdir()
            model = ModelArtifact(
                artifact_id="model-test",
                name="model-test",
                path=model_dir,
                meta={"base_model": "base-from-meta", "finetuning_type": "full"},
                producer="smoke_check",
            )
            artifact = run_eval_stage([model], sandbox, task=_NativeTask())[0]
            protocol = json.loads((artifact.path / "protocol.json").read_text())
            assert artifact.meta["mode"] == "lm_eval_native"
            assert artifact.meta["model"] == "vllm"
            assert artifact.meta["headline_score"] == 0.5
            assert captured["model"] == "vllm"
            assert captured["model_args"]["pretrained"] == "Qwen/Qwen2.5-7B-Instruct"
            assert captured["batch_size"] == "auto"
            assert captured["random_seed"] == 1
            assert captured["numpy_random_seed"] == 2
            assert captured["torch_random_seed"] == 3
            assert captured["fewshot_random_seed"] == 4
            assert protocol["resolved"]["lm_eval_model"] == "vllm"
            assert protocol["eval_config"]["gen_kwargs"]["max_gen_toks"] == 256
            assert (artifact.path / "samples-gsm8k.jsonl").exists()
    finally:
        lm_eval.simple_evaluate = original_simple_evaluate


def _check_legacy_artifact_and_helpers() -> None:
    assert EvalConfig(tasks=["gsm8k"]).normalized_mode() == "legacy_adapter"
    assert EvalConfig(tasks=["gsm8k"], mode="native").normalized_mode() == "lm_eval_native"

    with tempfile.TemporaryDirectory(prefix="dataos-eval-legacy-") as tmpdir:
        root = Path(tmpdir)
        model_dir = root / "adapter"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        model = ModelArtifact(
            artifact_id="model-lora",
            name="model-lora",
            path=model_dir,
            meta={"base_model": "Qwen/base", "finetuning_type": "lora"},
            producer="smoke_check",
        )
        args = _resolve_model_args(model, EvalConfig(tasks=["gsm8k"], mode="lm_eval_native"))
        assert args["pretrained"] == "Qwen/base"
        assert args["peft"] == str(model_dir)
        vllm_args = _resolve_model_args(
            model,
            EvalConfig(tasks=["gsm8k"], mode="lm_eval_native", model="vllm"),
            model_name="vllm",
        )
        assert vllm_args["pretrained"] == "Qwen/base"
        assert vllm_args["lora_local_path"] == str(model_dir)

        sandbox = Sandbox(root / "sandbox")
        sandbox.prepare()
        legacy_config = EvalConfig(tasks=["ifeval"])
        legacy_output = LmEvalRunOutput(
            tasks=["ifeval"],
            results={"ifeval": {"prompt_strict_acc": 0.25}},
            raw={"results": {"ifeval": {"prompt_strict_acc": 0.25}}},
            protocol={
                "mode": "legacy_adapter",
                "lm_eval_model": "DataOSLM",
                "adapter": "transformers",
            },
        )
        artifact = _eval_output_to_artifact(legacy_output, sandbox, model, legacy_config)
        assert artifact.meta["mode"] == "legacy_adapter"
        assert artifact.meta["model"] == "DataOSLM"
        assert artifact.meta["adapter"] == "transformers"
        assert artifact.meta["headline_score"] == 0.25

    context = AdapterCallContext(
        adapter=EvalAdapterConfig(
            name="transformers",
            config={"chat_template_already_applied": True},
        )
    )
    assert _render_chat_prompt(_Tokenizer(), "already-rendered", context) == "already-rendered"
    context = AdapterCallContext(adapter=EvalAdapterConfig(name="transformers", config={}))
    assert _render_chat_prompt(_Tokenizer(), "hello", context) == "TEMPLATE:hello"


def _check_posttrainbench_gsm8k_helpers() -> None:
    doc = {
        "question": "What is 40 + 2?",
        "answer": "Add them. #### 42",
    }
    prompt = posttrainbench_utils.doc_to_text(doc)
    assert "ANSWER: $ANSWER" in prompt
    assert "What is 40 + 2?" in prompt
    assert posttrainbench_utils.doc_to_target(doc) == "42"
    assert posttrainbench_utils.process_results(doc, ["Reasoning...\nANSWER: $42."])[
        "accuracy"
    ] == 1.0
    assert posttrainbench_utils.process_results(doc, ["Reasoning...\nANSWER: 41"])[
        "accuracy"
    ] == 0.0
    assert posttrainbench_utils.process_results(doc, ["Reasoning...\nANSWER: 4½"])[
        "accuracy"
    ] == 0.0


class _NativeTask:
    eval_config = {
        "mode": "lm_eval_native",
        "model": "vllm",
        "model_args": {
            "pretrained": "Qwen/Qwen2.5-7B-Instruct",
            "tensor_parallel_size": 1,
        },
        "tasks": ["gsm8k"],
        "num_fewshot": 8,
        "batch_size": "auto",
        "limit": 2,
        "gen_kwargs": {
            "temperature": 0.0,
            "do_sample": False,
            "max_gen_toks": 256,
        },
        "seed": [1, 2, 3, 4],
        "metadata": {"protocol_version": "test"},
        "log_samples": True,
    }


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "TEMPLATE:" + messages[0]["content"]


if __name__ == "__main__":
    main()
