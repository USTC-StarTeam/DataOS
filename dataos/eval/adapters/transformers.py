from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from dataos.core.artifact import ModelArtifact
from dataos.eval.types import AdapterCallContext, EvalRequest, ModelResponse


class TransformersAdapter:
    name = "transformers"

    def score_continuations(
        self,
        model_artifact: ModelArtifact,
        pairs: list[tuple[str, str]],
        config: AdapterCallContext,
    ) -> list[tuple[float, bool]]:
        if not pairs:
            return []
        model, tokenizer, torch = _load_generation_assets(model_artifact, config)
        device = _model_device(model)
        results: list[tuple[float, bool]] = []
        max_length = _model_max_length(model, tokenizer)
        try:
            for context, continuation in pairs:
                results.append(
                    _score_one_pair(
                        model=model,
                        tokenizer=tokenizer,
                        torch=torch,
                        device=device,
                        context=context,
                        continuation=continuation,
                        max_length=max_length,
                    )
                )
        finally:
            if hasattr(model, "cpu"):
                model.cpu()
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results

    def generate_until(
        self,
        model_artifact: ModelArtifact,
        requests: list[EvalRequest],
        config: AdapterCallContext,
    ) -> list[ModelResponse]:
        model, tokenizer, torch = _load_generation_assets(model_artifact, config)
        rendered_prompts = [
            _render_chat_prompt(tokenizer, request.prompt, config) for request in requests
        ]
        responses: list[ModelResponse] = []
        request_config = config.request
        if request_config is None:
            raise ValueError("AdapterCallContext.request is required for generation")
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
    config: AdapterCallContext,
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


def _render_chat_prompt(tokenizer: Any, prompt: str, config: AdapterCallContext) -> str:
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


def _model_max_length(model: Any, tokenizer: Any) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "max_position_embeddings", None) if config is not None else None,
        getattr(tokenizer, "model_max_length", None),
    ]
    for value in candidates:
        if isinstance(value, int) and 0 < value < 10**7:
            return int(value)
    return 4096


def _score_one_pair(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    context: str,
    continuation: str,
    max_length: int,
) -> tuple[float, bool]:
    """Score one (context, continuation) pair via a single forward pass.

    Tokenizes context with special tokens and continuation without, so the
    boundary token is always the first continuation token (matching how
    lm-eval's ``_encode_pair`` aligns logits to targets). Truncates the
    context from the left if (ctx + cont) would exceed the model's max
    position; never truncates the continuation.
    """
    cont_ids: list[int] = tokenizer(
        continuation, add_special_tokens=False
    )["input_ids"]
    if not cont_ids:
        return 0.0, True

    ctx_ids: list[int] = tokenizer(context, add_special_tokens=True)["input_ids"]
    if not ctx_ids:
        eos = tokenizer.eos_token_id
        bos = getattr(tokenizer, "bos_token_id", None)
        prefix = bos if isinstance(bos, int) else eos
        ctx_ids = [prefix] if isinstance(prefix, int) else []

    cont_len = len(cont_ids)
    available_ctx = max(1, max_length - cont_len)
    if len(ctx_ids) > available_ctx:
        ctx_ids = ctx_ids[-available_ctx:]

    ctx_len = len(ctx_ids)
    if ctx_len == 0:
        # Pathological: nothing to condition on; can't score the first token.
        return 0.0, False

    input_ids = torch.tensor([ctx_ids + cont_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[0]  # [seq_len, vocab]
    # logits at position i predict token i+1; continuation tokens occupy
    # positions [ctx_len, ctx_len + cont_len), so we need logits at
    # positions [ctx_len - 1, ctx_len + cont_len - 1).
    cont_logits = logits[ctx_len - 1 : ctx_len - 1 + cont_len, :]
    cont_target = torch.tensor(cont_ids, device=device, dtype=torch.long)

    log_probs = torch.log_softmax(cont_logits.float(), dim=-1)
    token_logprobs = log_probs.gather(-1, cont_target.unsqueeze(-1)).squeeze(-1)
    total_logprob = float(token_logprobs.sum().item())

    greedy_tokens = cont_logits.argmax(dim=-1)
    is_greedy = bool((greedy_tokens == cont_target).all().item())
    return total_logprob, is_greedy


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
