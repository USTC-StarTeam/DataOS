# 训练 + 评估 —— 开放-config 执行器

`train` 和 `eval` 都是**开放-config**:你通过 `config` dict 传任意 LLaMA-Factory / lm-eval 键,
原样转发 —— 没有白名单。DataOS 只占用它必须自己拥有的几个键、并校验必填项,所以你能拿到底层
工具的全部能力,而 DataOS 不挡道。

`config` 默认取本实验的 `spec["train"]` / `spec["eval"]`,所以 spec 设好后可以直接
`exp.train(ds)` / `exp.eval()` 不带参数。

## 训练 —— `exp.train(dataset, config=None) -> Model`

- **必填:** `model_name_or_path`、`template`。`finetuning_type` ∈ {`lora`, `full`, `freeze`}
  (默认 `lora`)。
- **保留键 —— DataOS 自己拥有,设置就报错:** `stage`、`do_train`、`dataset`、`eval_dataset`、
  `dataset_dir`、`output_dir`、`overwrite_output_dir`。
- **其余全部透传**到生成的 `train.yaml`:`learning_rate`、`num_train_epochs`、
  `per_device_train_batch_size`、`gradient_accumulation_steps`、`cutoff_len`、`lora_target`、
  `bf16`、`warmup_steps`…… 任意 LLaMA-Factory 键。
- **stub:** `config=None` 且无 `spec["train"]`,或 `{"mode": "stub"}` -> stub 训练(不碰
  GPU / torch)。真实训练需要 `train` extra + GPU。

```python
exp.train(ds, config={
    "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "template": "qwen",
    "finetuning_type": "lora", "lora_target": "all",
    "num_train_epochs": 1.0, "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8, "cutoff_len": 512,
    "learning_rate": 1e-4, "bf16": True,
})
```

返回 `Model`(path = 模型目录;`meta` 含 `base_model` / `template` / `finetuning_type` /
`mode` / `dataset_format` / `selected_rows`)。

## 评估 —— `exp.eval(model=None, config=None) -> Eval`

- `model` 默认取本实验训练出的模型;`config` 默认取 `spec["eval"]`。
- **stub 必须显式给:** 传 `config={"provider": "stub", "provider_config": {"score": <float>}}`
  (或放进 `spec["eval"]`)。光 `exp.eval()` 而没有 config 也没有 `spec["eval"]` 会**报错**
  —— 故意的:stub 评估需要一个分数,我们不替你编。(注意与训练的不对称:训练 `None` ⇒ stub。
  stub 训练无害;stub 评估会凭空造一个结果。)
- **真实 lm-eval**(需要 `eval` extra):

```python
exp.eval(config={"mode": "lm_eval_native", "model": "hf", "tasks": ["gsm8k"],
                 "limit": 50, "num_fewshot": 0, "apply_chat_template": True})
```

LoRA 模型自动解析为 `pretrained=<base>, peft=<adapter>`。返回 `Eval`(`score` = headline 分;
`metrics` = {"task::metric": value};在 `eval/` 下写 `results.json` 等)。
