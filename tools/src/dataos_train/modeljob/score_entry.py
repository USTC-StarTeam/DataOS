"""Deterministic per-sample scorer — runs INSIDE the model job (training env).

Invoked by the rendered ``run_score.sh`` as::

    python -m dataos_train.modeljob.score_entry --config modeljob.json

Reads the dataset, applies the model's own chat template, and computes a per-record
signal over the final assistant segment of each record (bf16, sdpa, no training):

- ``kind == "loss"``: per-token mean cross-entropy, batched; writes ``loss`` fields.
- ``kind == "gradient"``: LESS-style feature (arXiv 2402.04333) — completion-only CE
  gradient w.r.t. a temporary untrained LoRA adapter, sign-normalized, Rademacher
  projected (shared ``projection_seed``); writes ``score`` fields (cosine to the target
  direction when ``target`` is given, else the projected vector's L2 norm).

Writes ``scores.jsonl`` — one ``{"index", "key", "loss"|"score"}`` line per record —
into the cwd (the job's ``run_dir``).

torch/transformers/peft are imported lazily inside the scoring path, so importing this
module stays torch-free and the message-assembly helpers are unit-testable without the
training stack (same discipline as ``dataos_train.online``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataos_launch.io import atomic_write_text

from ..data.convert import read_records
from .backend import CONFIG_FILE, SCORES_FILE
from .gradient import cosine, flatten_grads, l2_norm, mean_direction, project, sign_normalize
from .spec import GradientInteraction, ModelJobSpec

_SHAREGPT_ROLES = {"human": "user", "gpt": "assistant", "system": "system"}


def detect_format(record: dict[str, Any]) -> str:
    """Per-record shape sniffing for ``dataset.format == 'auto'``."""
    if "messages" in record:
        return "messages"
    if "conversations" in record:
        return "sharegpt"
    return "alpaca"


def to_messages(record: dict[str, Any], fmt: str = "auto") -> list[dict[str, str]]:
    """Canonical record -> chat messages.

    alpaca: ``instruction`` (``+ "\n" + input`` when present) is the user turn, ``output``
    the assistant turn. messages: passed through. sharegpt: human/gpt/system mapped onto
    user/assistant/system.
    """
    fmt = detect_format(record) if fmt == "auto" else fmt
    if fmt == "messages":
        return [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in record.get("messages", [])
        ]
    if fmt == "sharegpt":
        return [
            {
                "role": _SHAREGPT_ROLES.get(str(c.get("from", "")), "user"),
                "content": str(c.get("value", "")),
            }
            for c in record.get("conversations", [])
        ]
    user = str(record.get("instruction", ""))
    if record.get("input"):
        user += "\n" + str(record["input"])
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(record.get("output", ""))},
    ]


def split_prompt(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    """Split into the prompt (everything before the final assistant turn) and its text.

    Only the final assistant segment is scored; earlier turns are context.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("records must end with an assistant turn to be loss-scorable")
    return messages[:-1], messages[-1].get("content", "")


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the shared token prefix — robust to tokenizer merges at the boundary."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _score(spec: ModelJobSpec) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    inter = spec.interaction
    if isinstance(inter, GradientInteraction):
        return _score_gradient(spec)
    records = read_records(spec.dataset.path)
    tok = AutoTokenizer.from_pretrained(spec.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = (
        AutoModelForCausalLM.from_pretrained(
            spec.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )

    out: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(records), inter.batch_size):
            batch = records[start : start + inter.batch_size]
            chunk = [to_messages(rec, spec.dataset.format) for rec in batch]
            prompts = [
                tok.apply_chat_template(
                    split_prompt(msgs)[0], tokenize=False, add_generation_prompt=True
                )
                for msgs in chunk
            ]
            fulls = [tok.apply_chat_template(msgs, tokenize=False) for msgs in chunk]
            enc = tok(
                fulls, return_tensors="pt", padding=True, truncation=True, max_length=inter.max_len
            ).to(device)
            # float() upcast matches HF's loss computation (ForCausalLMLoss), so the
            # numbers line up with what a training step would report.
            logits = model(**enc).logits.float()
            per_token = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                enc["input_ids"][:, 1:].reshape(-1),
                reduction="none",
            ).view(enc["input_ids"].size(0), -1)
            attn = enc["attention_mask"][:, 1:].bool()
            for i, prompt_text in enumerate(prompts):
                enc_prompt = tok(prompt_text, truncation=True, max_length=inter.max_len)
                prompt_ids = enc_prompt["input_ids"]
                full_ids = enc["input_ids"][i][: int(enc["attention_mask"][i].sum())].tolist()
                scored_from = max(_common_prefix_len(prompt_ids, full_ids) - 1, 0)
                mask = torch.zeros_like(per_token[i], dtype=torch.bool)
                mask[scored_from:] = attn[i][scored_from:]
                vals = per_token[i][mask]
                loss = float(vals.mean()) if vals.numel() else None
                out.append({"index": start + i, "key": inter.output_key, "loss": loss})
    return out


# Attention + MLP linear layers the temporary LoRA adapter attaches to (LESS setup).
_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _completion_ce(model: Any, tok: Any, record: dict[str, Any], fmt: str, max_len: int) -> Any:
    """Single-record completion-only CE (the same scored segment as the loss kind).

    Returns a scalar torch loss, or ``None`` when the assistant segment truncated away.
    """
    import torch

    device = next(model.parameters()).device
    msgs = to_messages(record, fmt)
    prompt_msgs, _ = split_prompt(msgs)
    prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(msgs, tokenize=False)
    enc = tok(full_text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    ids = enc["input_ids"]
    logits = model(**enc).logits.float()
    per_token = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1),
        reduction="none",
    ).view(ids.size(0), -1)
    prompt_ids = tok(prompt_text, truncation=True, max_length=max_len)["input_ids"]
    full_ids = ids[0][: int(enc["attention_mask"][0].sum())].tolist()
    scored_from = max(_common_prefix_len(prompt_ids, full_ids) - 1, 0)
    attn = enc["attention_mask"][0, 1:].bool()
    mask = torch.zeros_like(per_token[0], dtype=torch.bool)
    mask[scored_from:] = attn[scored_from:]
    vals = per_token[0][mask]
    return vals.mean() if vals.numel() else None


def _score_gradient(spec: ModelJobSpec) -> list[dict[str, Any]]:
    """LESS-style scoring: LoRA-gradient feature -> sign norm -> Rademacher projection.

    A temporary UNTRAINED LoRA adapter is attached to the frozen base model, so gradients
    are taken only w.r.t. the (small) LoRA parameters — aligned with LESS's LoRA
    gradients and cheap enough per record. One shared ``projection_seed`` covers every
    candidate AND target record (single projection matrix; per-record seeds would make
    the cosines incomparable).
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    inter = spec.interaction
    assert isinstance(inter, GradientInteraction)  # dispatched from _score
    records = read_records(spec.dataset.path)
    tok = AutoTokenizer.from_pretrained(spec.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        spec.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    base.requires_grad_(False)  # base model frozen; gradients hit LoRA params only
    model = get_peft_model(
        base,
        LoraConfig(r=inter.lora_r, target_modules=_LORA_TARGETS, task_type="CAUSAL_LM"),
    )
    model.train()  # enable grads (no training step is ever taken)

    def grad_feature(record: dict[str, Any], fmt: str) -> Any:
        """Projected gradient feature for one record; None when not scorable."""
        loss = _completion_ce(model, tok, record, fmt, inter.max_len)
        if loss is None:
            return None
        model.zero_grad(set_to_none=True)
        loss.backward()
        grads = [
            p.grad.detach().float().cpu().numpy()
            for p in model.parameters()
            if p.requires_grad and p.grad is not None
        ]
        flat = sign_normalize(flatten_grads(grads))
        return project(flat, inter.projection_dim, inter.projection_seed)

    target_dir = None
    if inter.target is not None:
        target_feats = [
            feat
            for rec in read_records(inter.target.path)
            if (feat := grad_feature(rec, inter.target.format)) is not None
        ]
        target_dir = mean_direction(target_feats)

    out: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        feat = grad_feature(rec, spec.dataset.format)
        if feat is None:
            out.append({"index": i, "key": inter.output_key, "score": None})
            continue
        score = cosine(feat, target_dir) if target_dir is not None else l2_norm(feat)
        out.append({"index": i, "key": inter.output_key, "score": score})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dataos_train.modeljob.score_entry")
    ap.add_argument("--config", default=CONFIG_FILE, help="path to the ModelJobSpec JSON")
    args = ap.parse_args(argv)
    spec = ModelJobSpec.model_validate(json.loads(Path(args.config).read_text(encoding="utf-8")))
    rows = _score(spec)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(SCORES_FILE, text)
    print(json.dumps({"scored": len(rows), "out": str(Path(SCORES_FILE).resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
