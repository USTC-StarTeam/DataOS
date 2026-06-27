# 训练 + 评估 —— 开放-config 执行器

> ⚠️ **过时(2026-06 agent-version 重构后)** —— 本文描述的是**已移除**的 in-process 执行器
> （`exp.train.run()` / `exp.eval.run()` 这类 Stage 对象、`EvalConfig` 的 `mode`/`provider`/`suite`）。
> 现在**只有一条执行路径**:主-agent 规划 → 每轮一个 sub-agent 在 H100 上完成
> `建数据(dataos_operator 算子)→ LLaMA-Factory 训练(Slurm)→ lm_eval_bridge 评估 → 写 report.json`。
> 训练超参由 sub-agent 写进 LLaMA-Factory YAML;评估固定为 `resources/code/posttrainbench/lm_eval_bridge`。
> 当前权威:**AGENTS.md / README.md**;sub-agent 手册:`resources/code/posttrainbench/PLAYBOOK.md`。本文仅作历史参考。

你通过 `config` dict 直接喂底层工具的参数,DataOS 基本不挡道。但**两者开放度不同**:

- **train 真透传**:任意 LLaMA-Factory 键原样写进 `train.yaml`,没有白名单(只拒下面那几个保留键)。
- **eval 是 `EvalConfig` 的字段集**(覆盖 lm-eval 常用键:`tasks` / `mode` / `limit` / `num_fewshot` /
  `batch_size` / `gen_kwargs` / `model_args` / `apply_chat_template` / `headline_metric` …)。不在其中的键会被
  **忽略并告警**(不再像以前那样直接 `TypeError`)。

`exp.train` / `exp.eval` 是 **Stage 对象**,`config` 来自 spec 的 `spec["train"]` / `spec["eval"]`,所以
spec 设好后直接 `exp.train.run()` / `exp.eval.run()` 即可,无需传参。要临时覆盖,改 `exp.train.config = {...}`
再 `run()`。两个 stage 都**自动取上游产物**(train 取 `exp.data` 的 `Data`,eval 取 `exp.train` 的 `Model`),
所以**不用手动传数据集或模型**。

## 训练 —— `exp.train.run() -> Model`

- **必填:** `model_name_or_path`、`template`。`finetuning_type` ∈ {`lora`, `full`, `freeze`}
  (默认 `lora`)。
- **保留键 —— DataOS 自己拥有,设置就报错:** `stage`、`do_train`、`dataset`、`eval_dataset`、
  `dataset_dir`、`output_dir`、`overwrite_output_dir`。
- **其余全部透传**到生成的 `train.yaml`:`learning_rate`、`num_train_epochs`、
  `per_device_train_batch_size`、`gradient_accumulation_steps`、`cutoff_len`、`lora_target`、
  `bf16`、`warmup_steps`…… 任意 LLaMA-Factory 键。
- **复现要设 `seed`:** 训练默认不确定。显式给 `seed`(开放-config 会透传给 LLaMA-Factory);DataOS 只把它
  **记录**进 manifest、**不自动注入**——没设就不可复现。
- **stub:** `spec["train"]` 为空,或 `{"mode": "stub"}` -> stub 训练(不碰 GPU / torch)。真实训练需要
  `train` extra + GPU。

```python
# 这些键放进 spec["train"];一次性覆盖也可 exp.train.config = {...}
exp.train.config = {
    "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "template": "qwen",
    "finetuning_type": "lora", "lora_target": "all",
    "num_train_epochs": 1.0, "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8, "cutoff_len": 512,
    "learning_rate": 1e-4, "bf16": True, "seed": 42,   # seed 为复现而设
}
exp.train.run()
```

返回 `Model`(一等 `Artifact`:`path` = 模型目录;`inputs` = 上游 `Data` 的 id = 血缘;`meta` 含
`base_model` / `template` / `finetuning_type` / `mode` / `dataset_format` / `selected_rows`)。
跑完后 `exp.train.summary()` / `exp.train.artifact` 拿到状态与产物。

## 评估 —— `exp.eval.run() -> Eval`

**"怎么评"是 runtime(你)的判断,不是 DataOS 的默认。** 场景定义评测(PostTrainBench 定义的是它仓库里
7 个 `src/eval/tasks/<task>/evaluate.py`,不是 lm-eval)——所以进入一个新场景的**第一步是评测 discovery**:
场景自带评分器/官方 harness 吗?bench 恰好在 lm-eval 任务库里吗?然后选执行通道:

- **lm-eval**(内置引擎,`mode` 缺省即 `lm_eval_native`,无需显式给;旧 legacy_adapter 桥已移除):
  bench 在 lm-eval 任务库里时最省(seeds / 任务级隔离 / per_task 全自动);端点模型走 lm-eval 自带的
  API backend(如 `local-chat-completions`)或 DataOS 的 tool-chat 插件。
- **`provider:"external"`**(万能接口):任何外部评分器 = 一条命令 + 一张 JSON 成绩单(见下节)。
- **stub**:冒烟。

**冻结纪律(discovery → 校验 → freeze → 复用)**:评测实现在 **project 开工时确定一次**——wrap 好后先拿
已知模型**校验对分**(和官方/论文数字对得上),把 harness 钉死(repo+commit / 包版本,填 `harness` 字段进
manifest),之后该 project 所有实验跑**同一个**冻结的 harness。**study 内协议不一致 = 排名无意义**,
`study.summary()` 会出 caution。

- **自动取上游 `exp.train` 的 `Model`**;`config` 取 `spec["eval"]`(或临时 `exp.eval.config = {...}`)。
- **eval-only(基线 / PostTrain 对照)**:没有上游 train 时,在 `spec["eval"]` 里给 `model_name_or_path`(或
  `base_model`,或 `model_args` 的 `pretrained=`)直接评一个**裸基座模型**——只 `exp.eval.run()`,不跑 data/train。
  用来先量 baseline、再和微调结果比收益(LoRA 实验自动 `pretrained=<base>,peft=<adapter>`,基线则只 `pretrained=`)。
- **stub 必须显式给:** `spec["eval"] = {"provider": "stub", "provider_config": {"score": <float>}}`。
  光 `exp.eval.run()` 而没有 config 也没有 `spec["eval"]` 会**报错**
  —— 故意的:stub 评估需要一个分数,我们不替你编。(注意与训练的不对称:训练 `None` ⇒ stub。
  stub 训练无害;stub 评估会凭空造一个结果。)
- **真实 lm-eval**(需要 `eval` extra):

```python
exp.eval.config = {"mode": "lm_eval_native", "model": "hf", "tasks": ["gsm8k"],
                   "limit": 50, "num_fewshot": 0, "apply_chat_template": True}
exp.eval.run()
```

LoRA 模型自动解析为 `pretrained=<base>, peft=<adapter>`。返回 `Eval`(一等 `Artifact`:`score` = headline 分;
`metrics` = {"task::metric": value};`inputs` = 上游 `Model` 的 id = 血缘;在 `eval/` 下写 `results.json` 等)。

### 外部 harness(`provider: "external"`)—— lm-eval 没有的 bench

Arena Hard / BFCL / HealthBench 这类自带 judge / 评测框架的 bench 不在 lm-eval 里 —— 用外部命令评,DataOS 只
负责**确定性地跑命令 + 收成绩单**(IoC:judge/harness 在外部,DataOS 不实现):

```python
exp.eval.config = {
    "provider": "external",
    # 占位符:{model_path}(待评模型/adapter 目录)、{base_model}、{output_dir}、{results_file}
    "command": "bash my_harness/run_bfcl.sh {model_path} {results_file}",
    "results_file": "external_results.json",   # 命令必须把 JSON 写到 {results_file}
    "timeout": 3600,                            # 可选;"env": {...} 可选(非密钥)
}
exp.eval.run()   # -> Eval;stdout/stderr tail 落 eval/,command 进 manifest(token 样式自动打码)
```

成绩单格式:`{"score": 0.61, "metrics": {"bfcl::overall": 0.61, ...}}`(`score` 缺省取 `headline_metric`
指定的 metric,否则取第一个)。**冻结时给 `"harness": {"repo": ..., "commit": ..., "judge_model": ...}`**——
原样记进 Eval meta 和 manifest,成绩单才能证明"是哪个评分器打的分"。**judge 需要的 LLM key / HF token 只走
env(harness 自己从环境读),绝不写进 command / config。** 诚实边界:外部 judge 不保证确定性 → `--replay`
容差自己定;DataOS 记录 command 与成绩单,不背 judge 的可复现性。

### 多 bench 套件(`suite`)—— 一个实验 = 整个场景的分

一次实验要在整个场景上打分(如 PostTrainBench 7 bench = lm-eval 4 任务 + 3 个外部 harness)时,把多个
eval 配置组合成一个 `suite`,**合并成一个 `Eval`**:

```python
spec["eval"] = {"suite": [
    {"name": "lm",     "mode": "lm_eval_native", "model": "hf",
     "tasks": ["gsm8k", "aime25", "gpqa_diamond_zeroshot", "humaneval"],
     "limit": 50, "apply_chat_template": True, "confirm_run_unsafe_code": True},
    {"name": "bfcl",   "provider": "external", "command": "python .../bfcl/evaluate.py --model-path {model_path} --json-output-file {results_file}",
     "harness": {"repo": "aisa-group/PostTrainBench", "commit": "<sha>"}},
    {"name": "arena",  "provider": "external", "command": "..."},
    {"name": "health", "provider": "external", "command": "..."},
]}
exp.eval.run()
```

每条 entry 走各自的正常路径(lm-eval 的 seeds/隔离、external 的收据全保留),产物落 `eval/<entry名>/`;
合并结果:`metrics` 取并集、`per_task` 跨全部 bench、**`macro_average` = 全场景宏平均**(report 顶层 `score`、
`study.summary()` 排名都用它)、某条 entry 挂了进 `skipped_tasks` 不连累其余、每条的收据(command/harness/分)
留在 `eval.suite`。eval-only 基线照常:顶层给 `model_name_or_path` 即可。不支持嵌套 suite。

> **一个 task 可能出多个指标**(如 gsm8k → `strict-match` 和 `flexible-extract`)。headline 默认取第一个;
> 想固定用哪个,在 eval config 里加 `"headline_metric": "flexible-extract"`。`study.summary()` 会列出每个实验的
> 全部 `metrics`,并在 headline 冠军和某个指标冠军不一致时给出 `caution`——**别只看 headline**,结论可能因指标而翻转。

> **多 bench 一次评**(`tasks=[...]`):`report.json` 顶层 `score` 与 `eval.macro_average` = 各 bench 主指标的
> **宏平均**(`study.summary()` 按它排名),`eval.per_task` 给每个 bench 的分。三个坑:① 某 bench 数据**被门控/缺失**
> (如 GPQA 的 `Idavidrein/gpqa`)只会被**跳过**并记入 `eval.skipped_tasks`,不再连累其它 bench(要真测门控
> bench:HF 网页接受条款 + `HF_TOKEN` 临时进 env,见 `remote.md`;token 绝不进 spec/manifest);② **长生成任务**
> (如 `aime25`,task 默认 `max_gen_toks=32768`)必须在 `gen_kwargs` 里压 `max_gen_toks=1024` 之类,否则极慢爆时间;
> ③ 代码 bench(`humaneval`/`mbpp`)给 `confirm_run_unsafe_code=True` 即可,DataOS 会自动置 `HF_ALLOW_CODE_EVAL=1`;
> ④ **小样本 smoke 只作"没崩"检查,不能用于实验排名**(实测 n=20 把 0.845/0.76/0.68 的真实分化抹平成三方平局)
> ——结论在 formal 量级出,且判停 margin 必须大于宏平均的测量噪声(否则判停门是掷硬币)。

> 日志里出现 `fatal: not a git repository ... Stopping at filesystem boundary` 是 lm-eval / llamafactory 探
> git hash 的无害噪音,忽略即可。
