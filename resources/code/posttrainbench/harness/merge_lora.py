#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model so vllm-based harnesses can load it.

Usage: merge_lora.py <adapter_dir> <merged_out_dir>
(The base model id is read from adapter_config.json; runs in the PostTrainBench venv,
which has transformers+peft. ~1 min for a 1.7B model.)"""
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_dir, out_dir = sys.argv[1], sys.argv[2]
base = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))["base_model_name_or_path"]
print(f"[merge] base={base} adapter={adapter_dir}")
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
model.save_pretrained(out_dir)
AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
print(f"[merge] MERGED -> {out_dir}")
