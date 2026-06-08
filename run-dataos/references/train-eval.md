# 训练 + 评估 —— 开放-config 执行器

你通过 `config` dict 直接喂底层工具的参数,DataOS 基本不挡道。但**两者开放度不同**:

- **train 真透传**:任意 LLaMA-Factory 键原样写进 `train.yaml`,没有白名单(只拒下面那几个保留键)。
- **eval 是 `EvalConfig` 的字段集**(覆盖 lm-eval 常用键:`tasks` / `mode` / `limit` / `num_fewshot` /
  `batch_size` / `gen_kwargs` / `model_args` / `apply_chat_template` / `headline_metric` …)。不在其中的键会被
  **忽略并告警**(不再像以前那样直接 `TypeError`)。

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

> **一个 task 可能出多个指标**(如 gsm8k → `strict-match` 和 `flexible-extract`)。headline 默认取第一个;
> 想固定用哪个,在 eval config 里加 `"headline_metric": "flexible-extract"`。`study.compare()` 会列出每个实验的
> 全部 `metrics`,并在 headline 冠军和某个指标冠军不一致时给出 `caution`——**别只看 headline**,结论可能因指标而翻转。

> 日志里出现 `fatal: not a git repository ... Stopping at filesystem boundary` 是 lm-eval / llamafactory 探
> git hash 的无害噪音,忽略即可。
