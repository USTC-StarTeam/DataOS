#!/usr/bin/env python
"""Self-contained LoRA SFT trainer for the H100 Slurm remote-train backend.

Runs inside the prebuilt env at /data/tangjiahui/dataos_train/.venv on an H100 gpu node
(offline; base model + data are staged locally). Reads an alpaca-style jsonl
({"instruction","input","output"}; "messages"/"conversations" also accepted) and writes a
PEFT LoRA adapter to --output-dir (adapter_config.json + adapter_model.safetensors), which
the caller rsyncs back and DataOS treats as a finetuning_type=lora Model.
"""

from __future__ import annotations

import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def _row_to_text(ex: dict, tok) -> str:
    # messages / sharegpt → chat template; else alpaca instruction/input/output
    msgs = ex.get("messages") or ex.get("conversations")
    if msgs:
        norm = []
        for m in msgs:
            role = m.get("role") or {"human": "user", "gpt": "assistant"}.get(m.get("from"), m.get("from"))
            norm.append({"role": role, "content": m.get("content") or m.get("value") or ""})
        try:
            return tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=False)
        except Exception:
            return "\n".join(m["content"] for m in norm)
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    prompt = f"{instr}\n\n{inp}" if inp else instr
    return f"{prompt}\n{out}{tok.eos_token}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--data", required=True, help="training jsonl")
    ap.add_argument("--output-dir", required=True, help="adapter output dir")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=args.data, split="train")
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, ds.num_rows)))
    ds = ds.map(lambda ex: {"text": _row_to_text(ex, tok)}, remove_columns=ds.column_names)
    ds = ds.map(
        lambda ex: tok(ex["text"], truncation=True, max_length=args.max_len),
        remove_columns=["text"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=f"{args.output_dir}/_trainer",
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()

    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    meta = {"base_model": args.base_model, "finetuning_type": "lora",
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}
    with open(f"{args.output_dir}/dataos_train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("SAVED_ADAPTER", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
