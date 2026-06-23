"""SFT a base model on a prepared {prompt,target,text} dataset.

``--finetuning-type lora`` (default) trains LoRA adapters then merges them into the base to
produce ``--merged-dir`` (final_model); ``--finetuning-type full`` trains all parameters and
saves the result directly as ``--merged-dir``. The method + hyperparameters are chosen by the
main-agent (planner) per experiment; this script just executes them."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from pathlib import Path

# The shared `train` conda env ships an autoawq that is incompatible with its transformers
# (4.57.x removed transformers.activations.PytorchGELUTanh), and peft's LoRA injection eagerly
# imports the awq integration. We don't use AWQ, so hide it from package detection BEFORE
# importing transformers/peft. In-process only — does NOT touch the shared env.
import importlib.util as _ilu

_ilu_find_spec = _ilu.find_spec


def _hide_awq(name, *args, **kwargs):
    if name == "awq" or name.startswith("awq."):
        return None
    return _ilu_find_spec(name, *args, **kwargs)


_ilu.find_spec = _hide_awq

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--merged-dir", required=True)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--finetuning-type", default="lora", choices=["lora", "full"],
                   help="lora = LoRA adapters then merge; full = full-parameter SFT (no merge)")
    p.add_argument("--lora-r", type=int, default=64, help="LoRA rank (ignored for --finetuning-type full)")
    p.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha (ignored for full)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[train] base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[train] loading dataset from {args.data_dir}")
    ds = load_from_disk(args.data_dir)
    print(f"[train] dataset size: {len(ds)}")

    def tokenize(ex):
        text = ex["text"] + tok.eos_token
        out = tok(text, truncation=True, max_length=args.max_len, padding=False)
        # Mask prompt tokens in labels so loss is only on completion
        prompt_ids = tok(ex["prompt"] + " ", truncation=True, max_length=args.max_len, padding=False)["input_ids"]
        p_len = len(prompt_ids)
        labels = list(out["input_ids"])
        for i in range(min(p_len, len(labels))):
            labels[i] = -100
        out["labels"] = labels
        return out

    print("[train] tokenizing")
    ds_tok = ds.map(tokenize, remove_columns=ds.column_names, num_proc=8)
    print(f"[train] tokenized; example token count: {len(ds_tok[0]['input_ids'])}")

    print("[train] loading base model in bf16")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    if args.finetuning_type == "lora":
        print("[train] adding LoRA adapters")
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        print("[train] FULL-parameter fine-tuning — all weights trainable (no LoRA, no merge)")

    # pads input_ids AND the pre-masked labels (-100) to the batch max — handles variable
    # lengths; DataCollatorForLanguageModeling left pre-set labels unpadded (can't batch).
    collator = DataCollatorForSeq2Seq(tokenizer=tok, padding=True, label_pad_token_id=-100)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        save_total_limit=1,
        report_to=[],
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        optim="adamw_torch",
        max_grad_norm=1.0,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok,
        data_collator=collator,
    )
    print("[train] starting training")
    trainer.train()

    Path(args.merged_dir).mkdir(parents=True, exist_ok=True)
    if args.finetuning_type == "lora":
        print("[train] done; saving adapter + merging LoRA into base for final_model")
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)
        del model, trainer
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, args.output_dir)
        merged = merged.merge_and_unload()
        merged.save_pretrained(args.merged_dir, safe_serialization=True)
        tok.save_pretrained(args.merged_dir)
        print(f"[train] merged model written to {args.merged_dir}")
    else:
        # full-parameter SFT: the trained model IS the final model — save it directly, no merge
        print("[train] done; saving full fine-tuned model as final_model")
        trainer.save_model(args.merged_dir)
        tok.save_pretrained(args.merged_dir)
        print(f"[train] full model written to {args.merged_dir}")


if __name__ == "__main__":
    main()
