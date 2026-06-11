# PostTrainBench 官方评分器 —— 冻结 harness(DataOS external/suite 资产)

**协议 = PostTrainBench 官方 `src/eval/tasks/<bench>/evaluate.py`**(7 个 bench 同一契约
`--model-path/--limit/--json-output-file`),经 `run_ptb.sh` 包成 DataOS `provider:"external"`
命令:自动 merge LoRA、设镜像/代理 env、跑官方评分器、把平面 metrics 转成 `{score, metrics}` 收据。

## 冻结身份(写进每个 entry 的 `harness` 字段)

```json
{"repo": "aisa-group/PostTrainBench", "commit": "d3496fa7", "patch": "judge-model-env",
 "inspect_evals": "bbc1e495", "vllm": "0.11.0", "transformers": "4.57.3",
 "judge_model": "mimo-v2.5-pro (经 clash 代理;分数不可对标官方榜,仅内部排名)"}
```

部署:harness 环境在 a800 `/data/hcgu/dev/PostTrainBench`(独立 venv,`PINNED_COMMITS.txt` 为收据;
判据 key 在 `~/.mimo_key`,只经 env 注入)。环境变更只归主 agent。

## suite 配置(2026-06-11 实测校准:Qwen3-1.7B 基线 bfcl 0.85 / gsm8k 0.85,各 ~200s@limit20)

`W=/data/hcgu/dev/DataOS/run-dataos/harnesses/posttrainbench/run_ptb.sh`;烟测/正式只差 limit 与 timeout。

| entry | command(smoke) | timeout | 正式 limit/timeout | 备注 |
|---|---|---|---|---|
| ptb_gsm8k | `bash $W gsm8k {model_path} {results_file} 20` | 600 | 200 / 1800(全量 1319 ≈3600) | 10-shot ANSWER 协议,基线≈0.85,**与 lm-eval gsm8k 不可比** |
| ptb_bfcl | `bash $W bfcl {model_path} {results_file} 20` | 600 | -1 / 10800 | 规则判分,无 judge;`all2` 实为 stderr(转换器已处理) |
| ptb_humaneval | `bash $W humaneval {model_path} {results_file} 10` | 2400 | 150 / 10800 | `sandbox="local"` 裸执行生成代码;max_connections=1 慢 |
| ptb_aime2025 | `bash $W aime2025 {model_path} {results_file} 5` | 2400 | -1(仅30题)/ 10800 | 16k token 长 CoT,**单独校准,勿沿用上面边际成本** |
| ptb_healthbench | `bash $W healthbench {model_path} {results_file} 3` | 2400 | 32 / 10800 | 需 judge(~/.mimo_key);rubric 加权分 |
| ptb_arenahardwriting | `bash $W arenahardwriting {model_path} {results_file} 3` | 2400 | 32 / 10800 | 需 judge;winrate vs vendored o3-mini 答案 |
| ptb_gpqamain | — | — | — | **暂缓**:HF 门控数据集(`Idavidrein/gpqa`),需用户 HF_TOKEN+接受条款 |

spec 形如:
```python
{"name": "ptb_gsm8k", "provider": "external", "timeout": 600,
 "command": "bash /data/hcgu/dev/DataOS/run-dataos/harnesses/posttrainbench/run_ptb.sh gsm8k {model_path} {results_file} 20",
 "harness": {<上面的冻结身份>}}
```

## 注意
- **一次只跑一个 bench / 一张卡**(每个都自起 vllm);launch 时钉 `CUDA_VISIBLE_DEVICES`。
- **smoke 只是"没崩"检查,不能用于排名**(实测:smoke 下 bfcl 三方平 0.85,formal 下 0.845/0.76/0.68
  全分开)。结论一律在 formal limit 出;判停门的 margin 要和宏平均的测量噪声比(n=200/200/100 时
  macro se≈0.018——小于它的差异是掷硬币,不是信号)。
- **eval-only 基线**:`spec = {"name": "baseline", "eval": {"suite": [...], "model_name_or_path": "Qwen/Qwen3-1.7B"}}`
  ——只 `exp.eval.run()`;`{model_path}` 解析为 HF id,vllm 经镜像加载。
- **prompting 形态**(评测协议的一部分):所有 bench 经 vllm 套用 `templates/qwen3.jinja` chat template;
  gsm8k 为 10-shot + 显式 `ANSWER:` 指令;thinking 模式未被显式禁用(已知未知,对比实验间一致即可)。
- LoRA merge 缓存默认在 `~/.cache/dataos_ptb_merged/<内容hash>`(按 adapter 权重**内容**取键——原地重训
  绝不会复用旧 merge;`PTB_MERGE_CACHE` 可改),不写 artifact 目录;不再信任旧的 in-artifact `_merged`。
- judge 类(arena/health)分数 judge-dependent:study 内同 judge 才可比;`--replay` 容差放宽。
- aime2025 / humaneval 的 smoke 上限是估计值(校准只实测了 bfcl/gsm8k),首跑留余量。
