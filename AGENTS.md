# AGENTS.md

## 这是什么仓库

`DataOS` —— 本地优先的 SFT 数据流水线**确定性执行器库**。设计刻意很窄:

- DataOS **不调用 LLM**:加载、处理、训练、评估、复盘都是本地 Python。规划和判断由 **Claude Code**
  (runtime / planner)在库外完成。
- 两个容器:`Study`(多实验,就是一个目录)→ `Experiment`(单次)。`compare()` 读盘聚合,所以并行与否
  由 runtime 决定。
- 数据集是外部 `dataos_operator.Dataset`;算子来自外部 `dataos-operator` 包,不在本仓库。
- `train` 真透传任意 LLaMA-Factory 键(只拒保留键);`eval` 用 `EvalConfig` 字段集(未知键忽略 + 告警)。
- 没有内嵌 agent、没有 `run()`、没有调度器、没有研究层 / 守护进程 / 项目队列。

在本仓库工作的 agent,除非用户明确要求,应保持这些约束。

## 技术栈

- Python `>=3.11`;`uv` + `setuptools`;`ruff`。
- 核心依赖:`datasets`、`pyarrow`、`dataos-operator`(git)。Extras:`train`(llamafactory + torch)、
  `eval`(lm-eval)、`vllm`。

## 仓库结构

- `dataos/_catalog.py` —— `catalog()` 算子发现
- `dataos/loading.py` —— `load()` 数据加载(hf:// / 本地文件 / save_to_disk 目录)
- `dataos/_experiment.py` —— `Experiment` / `Model` / `Eval` / `experiment()`
- `dataos/_study.py` —— `Study` / `study()`
- `dataos/workspace.py`、`dataos/report.py`、`dataos/_env.py`、`dataos/cli.py`
- `dataos/train/llamafactory.py` —— 开放-config 训练执行器
- `dataos/eval/` —— lm-eval 桥接(runner / config / types / backend / adapters / models)
- `run-dataos/` —— 给 Claude Code 的 skill(怎么驱动这个库)
- `docs/` —— 中文说明(实验布局、算子)

## 本地命令

```bash
uv sync --extra dev                          # 装依赖
uv run ruff check .                          # lint
python run-dataos/scripts/doctor.py    # 环境自检
```

## 远程 GPU 主机

需要真实训练 / 评测时:

- **`a800-2-chaobo`(主力开发机)** —— `HostName 210.45.70.162`,`Port 1743`,`User hcgu`,
  7× A800 80GB,`HOME=/data/hcgu`,开发目录在 `/data/hcgu/dev`(从本地 rsync)。GitHub 被墙,机上有
  clash 代理(`127.0.0.1:7890`),HF 用 `HF_ENDPOINT=https://hf-mirror.com`;SSH 走 Bitwarden SSH agent。
- 其他可用主机:`bdaa_5090-1`(`114.214.255.119:1108`,3× RTX 5090)、`bdaa_5090-2`(`:1109`,
  2× RTX 5090)、`hcgu_h100`(实验室 H100 Slurm 集群;登录节点仅编排,GPU 作业走 Slurm)。

操作规则:

- **不要把 token / API key / 私钥写进任何 run manifest 或落盘文件。**
- 远程只当执行后端;把 DataOS 产物(`report.json` / `review.json`)拉回仓库 workspace,或确认已落在远端
  持久目录后,再算这次跑完成。
- 别假设远端依赖和本地一致;长跑前先 `doctor.py` 核对 Python / CUDA / extras。
- 真实训练前把 venv 的 bin 加进 PATH(`PATH=.venv/bin:$PATH`),否则 `llamafactory-cli` 找不到。
- 从干净目录(如 `/tmp`)跑脚本,避免 cwd 里的 `./dataos` 盖掉安装的包。

## 必须遵守的运行时事实

- **数据**:外部 `dataos_operator.Dataset`(HF `save_to_disk` 背书)。处理无 DataOS API,直接
  `Op(**p).run(ds)`。
- **训练开放-config**:必填 `model_name_or_path` + `template`;`finetuning_type` ∈ {lora, full, freeze};
  保留键(`stage` / `do_train` / `dataset` / `eval_dataset` / `dataset_dir` / `output_dir` /
  `overwrite_output_dir`)由 DataOS 写,设置即报错;其余透传到 `train.yaml`。
- **评估**:lm-eval 桥接,config = `EvalConfig` 字段集(未知键忽略 + 告警);stub 必须显式 `{"provider": "stub", ...}`。
- **落盘**:`<workspace>/<study_id>/{plan.json, review.json, experiments/<name>/{dataset,model,eval,logs,_lf}/ + report.json}`。`compare()` 只读盘。

## 改动指引

- 算子在外部 `dataos-operator` 包,不在本仓库;别再加内置算子注册表。
- 别把 `run()` / 计划 agent / LLM 客户端加回来——判断属于 Claude。
- train 保持真开放-config(别加白名单);eval 的 `EvalConfig` 是有意的字段集——要扩展就加字段,别改成任意 key。
- 不维护 pytest;除非用户明确要求,别重建测试套件。

## 校验

```bash
uv run ruff check .
```

有 GPU 时按需跑 stub 或真实冒烟(见 skill)。

## 代码与文档风格

- 跟随现有 Python 风格;行宽 ≤ 100(ruff)。保留类型注解与 dataclass 写法。
- 注释稀疏,只解释非显然逻辑。
- **文档用中文**(本仓库面向中文团队);代码标识符、API 名、配置键保持英文。

## Feishu / Lark CLI 注意

- 操作用户的飞书资源时别依赖沙箱执行——沙箱线程可能访问不到 `lark-cli` 用的本地 keychain(表现为
  `keychain not initialized`,即便已授权)。必要时用绝对路径直接跑 Node 入口,别依赖 `PATH` 里有 Homebrew。
- wiki URL 先把 wiki 节点解析成 `obj_token`,再操作底层 doc token。
