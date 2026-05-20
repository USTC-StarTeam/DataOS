from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.run_config).read_text(encoding="utf-8"))
    model_path = resolve_cached_model_path(config["base_model"]["remote_cache_hint"])
    source_path = (
        Path(config["fixed_pool"]["remote_sharegpt_root"]).expanduser().resolve() / args.source
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Fixed-pool source not found: {source_path}")

    seed_everything(args.seed)
    raw_dataset = load_source_dataset(source_path)
    selected = select_dataset(raw_dataset, max_samples=args.max_samples, seed=args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted = selected.map(
        lambda example: {"text": format_sharegpt_example(example, tokenizer)},
        remove_columns=list(selected.column_names),
        desc="format_sharegpt",
    )
    tokenized = formatted.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_seq_length,
        ),
        batched=True,
        remove_columns=["text"],
        desc="tokenize",
    )
    tokenized = tokenized.filter(lambda row: len(row["input_ids"]) > 8, desc="filter_short")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    recipe = {
        "run_config": str(Path(args.run_config).resolve()),
        "base_model": config["base_model"]["hf_id"],
        "resolved_model_path": str(model_path),
        "task": config["task"]["posttrainbench_id"],
        "fixed_pool_source": args.source,
        "fixed_pool_source_path": str(source_path),
        "max_samples": args.max_samples,
        "effective_samples_after_filter": len(tokenized),
        "max_steps": args.max_steps,
        "max_seq_length": args.max_seq_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "loss_scope": "full_sequence_causal_lm",
        "purpose": "smoke trial for fixed-pool agent/base/task execution, not a paper result",
    }
    (output_dir / "training_recipe.json").write_text(
        json.dumps(recipe, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_data_manifest(output_dir / "data_manifest.json", selected, source_path, args)

    train_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_state"),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim="adamw_torch",
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": train_args,
        "train_dataset": tokenized,
        "data_collator": collator,
    }
    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    result = trainer.train()

    final_model_dir = output_dir / "final_model"
    adapter_dir = final_model_dir / "lora_adapter"
    merged_dir = final_model_dir / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(merged_dir)

    summary = {
        "train_result": result.metrics,
        "adapter_dir": str(adapter_dir),
        "merged_model_dir": str(merged_dir),
        "tokenized_rows": len(tokenized),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "eval_log.md").write_text(
        "# Eval Log\n\nFinal GSM8K lm-eval is launched by the Slurm wrapper after training.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small LoRA SFT trial from fixed pool.")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source", default="numinamath_cot")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.set_defaults(gradient_checkpointing=True)
    return parser.parse_args()


def resolve_cached_model_path(cache_hint: str) -> Path:
    hint = Path(cache_hint).expanduser()
    snapshots = hint / "snapshots"
    if snapshots.exists():
        candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
        if candidates:
            return candidates[-1].resolve()
    if hint.exists():
        return hint.resolve()
    raise FileNotFoundError(f"Could not resolve model cache path: {cache_hint}")


def load_source_dataset(source_path: Path) -> Dataset:
    dataset = load_from_disk(str(source_path))
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        first_split = next(iter(dataset))
        return dataset[first_split]
    if isinstance(dataset, Dataset):
        return dataset
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def select_dataset(dataset: Dataset, *, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset.shuffle(seed=seed)
    return dataset.shuffle(seed=seed).select(range(max_samples))


def format_sharegpt_example(example: dict[str, Any], tokenizer: Any) -> str:
    messages: list[dict[str, str]] = []
    system = str(example.get("system") or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    for turn in normalise_conversations(example.get("conversations")):
        role = normalise_role(turn.get("from") or turn.get("role"))
        content = str(turn.get("value") or turn.get("content") or "").strip()
        if role and content:
            messages.append({"role": role, "content": content})
    if not messages:
        return ""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        rendered: list[str] = []
        for message in messages:
            rendered.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>")
        return "\n".join(rendered) + "\n"


def normalise_conversations(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return decoded if isinstance(decoded, list) else []
    if isinstance(value, list):
        return [turn for turn in value if isinstance(turn, dict)]
    return []


def normalise_role(value: Any) -> str | None:
    role = str(value or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    return None


def write_data_manifest(path: Path, selected: Dataset, source_path: Path, args: argparse.Namespace) -> None:
    row_ids: list[Any] = []
    for index in range(min(len(selected), 20)):
        row = selected[index]
        row_ids.append(
            {
                "selected_index": index,
                "source_row_index": row.get("source_row_index"),
                "raw_id": row.get("raw_id"),
                "source_id": row.get("source_id"),
            }
        )
    manifest = {
        "fixed_pool_source_path": str(source_path),
        "source": args.source,
        "selection": "deterministic_shuffle_prefix",
        "seed": args.seed,
        "max_samples": args.max_samples,
        "selected_rows": len(selected),
        "first_selected_rows": row_ids,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
