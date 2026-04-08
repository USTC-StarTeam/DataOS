from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from dataos.core.artifact import ModelArtifact
from dataos.eval.types import EvalConfig, EvalRequest, ModelResponse


class TransformersAdapter:
    name = "transformers"

    def generate_until(
        self,
        model_artifact: ModelArtifact,
        requests: list[EvalRequest],
        config: EvalConfig,
    ) -> list[ModelResponse]:
        model, tokenizer, torch = _load_generation_assets(model_artifact, config)
        rendered_prompts = [
            _render_chat_prompt(tokenizer, request.prompt, config) for request in requests
        ]
        responses: list[ModelResponse] = []
        request_config = config.request
        if request_config is None:
            raise ValueError("EvalConfig.request is required for generation")
        generate_kwargs = {
            "max_new_tokens": request_config.max_new_tokens,
            "do_sample": request_config.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        try:
            for start in range(0, len(rendered_prompts), request_config.batch_size):
                batch_requests = requests[start : start + request_config.batch_size]
                batch_prompts = rendered_prompts[start : start + request_config.batch_size]
                tokenized = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                device = _model_device(model)
                tokenized = {key: value.to(device) for key, value in tokenized.items()}
                with torch.inference_mode():
                    outputs = model.generate(**tokenized, **generate_kwargs)
                generated_ids = outputs[:, tokenized["input_ids"].shape[1] :]
                decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for request, text in zip(batch_requests, decoded):
                    responses.append(
                        ModelResponse(
                            sample_id=request.sample_id,
                            index=request.index,
                            request_type=request.request_type,
                            text=text.strip(),
                        )
                    )
        finally:
            if hasattr(model, "cpu"):
                model.cpu()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return responses



def _load_generation_assets(
    model_artifact: ModelArtifact,
    config: EvalConfig,
):
    if importlib.util.find_spec("torch") is None:
        raise ValueError("PyTorch is required for native evaluation")
    if importlib.util.find_spec("transformers") is None:
        raise ValueError("transformers is required for native evaluation")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config = config.adapter
    trust_remote_code = False
    if adapter_config is not None:
        trust_remote_code = bool(adapter_config.config.get("trust_remote_code", False))

    model_path = Path(model_artifact.path)
    base_model = model_artifact.meta.get("base_model")
    tokenizer_source = str(base_model or model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    peft_config_path = model_path / "adapter_config.json"
    if peft_config_path.exists():
        if importlib.util.find_spec("peft") is None:
            raise ValueError("peft is required to evaluate LoRA adapters")
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        model_source = str(base_model or model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    model.eval()
    return model, tokenizer, torch


def _render_chat_prompt(tokenizer: Any, prompt: str, config: EvalConfig) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        adapter_config = config.adapter.config if config.adapter is not None else {}
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "enable_thinking" in adapter_config:
            template_kwargs["enable_thinking"] = bool(adapter_config["enable_thinking"])
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                **template_kwargs,
            )
        except TypeError:
            template_kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                **template_kwargs,
            )
    return prompt


def _model_device(model: Any):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for mapped in hf_device_map.values():
            if isinstance(mapped, str) and mapped not in {"cpu", "disk"}:
                return mapped
    return "cpu"
