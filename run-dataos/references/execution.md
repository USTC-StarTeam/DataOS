# 执行 —— 运行模式与并行

串行还是并行,是**你**的事,不归 DataOS。project / study 都是目录,`study.summary()` 读每个实验盘上的
`report.json`,所以无论你怎么驱动,SDK 都一样。挑合适的模式。

## 光谱

```
一个大脚本(全内存)     ↔     一实验一进程      ↔     一阶段一调用
   批处理、最快                自适应 / 并行甜点区          最灵活
```

状态在调用之间靠两种方式传:

- **同进程(一个脚本):** 内存里的 `project` / `study` / `exp` / `stage` 对象顺序流转。最简单 —— 适合计划
  已固定的情形。
- **跨进程(分开跑):** 文件系统是 join。`dataos.project(id=...)` 是 **open-or-create** —— 新进程
  重开 project、`project.load()` 把整个项目(所有 study + experiment)**回灌成对象**(状态 / 血缘 / 产物全恢复),
  再 `run()` 即**续跑**(`completed` 且 config 未变的跳过;pending / failed / `stale`(上游重跑过)/ config 漂移
  的会重跑且**清空产物目录**;还活着的 `running` 拒绝再驱动防双跑;`run(force=True)` 无条件重跑)。结果在 `report.json` 里。这就是
  "中途停下来判断""换机接着跑"和并行能成立的原因。没有 DataOS session 或守护进程 —— 状态要么在内存、要么在磁盘。

## 串行(一个进程)

```python
study = dataos.project(id="qwen-gsm8k").study(id="lr-sweep"); study.set_plan(plan)
for spec in study.plan["experiments"]:
    exp = study.experiment(spec)
    ds = build_dataset(spec)        # 你的配方(见 data-prep.md);声明式则用 spec["data"]+exp.data.run()
    exp.data.set(ds).run(); exp.train.run(); exp.eval.run()   # report.json 自动落
study.summary(); study.review(conclusion="…")
```

## 并行(一实验一进程,各钉一张卡)

自然单位是**一实验一进程**,因为 `data -> train -> eval -> report` 共享内存里的 stage 产物
(`exp.data.artifact` / `exp.train.artifact`)—— 把 train 和 eval 拆到不同进程时,EvalStage 会自动从盘上的
`train.json` 重建 `Model`(血缘照样连上)。把共享数据集准备并保存一次,然后用自带的
`scripts/run_one.py`:

```bash
python -c "import dataos; dataos.project(id='qwen-gsm8k').study(id='lr-sweep').set_plan(__import__('json').load(open('plan.json')))"
# …… 用算子构建数据集,然后 ds.save_to_disk('prepared/') ……
CUDA_VISIBLE_DEVICES=6 python scripts/run_one.py --project qwen-gsm8k --study lr-sweep --spec lr-1e-4 --data prepared/ &
CUDA_VISIBLE_DEVICES=5 python scripts/run_one.py --project qwen-gsm8k --study lr-sweep --spec lr-5e-5 --data prepared/ &
wait
python -c "import dataos, json; print(json.dumps(dataos.project('qwen-gsm8k').study('lr-sweep').summary(), ensure_ascii=False, indent=2))"
```

DataOS 既不调度、也不知道并行 —— 分卡和并发度完全是你的事。`summary()` 只是 glob 现有的
`report.json`,所以容忍还在跑或缺失的实验。

> 在 agent harness(Claude Code)里驱动时:**一条命令只发射一个 `run_one.py`**,N 个并行 = N 条独立命令
> (远程则 N 条独立 ssh)。含 `&` 的整条命令会被 harness 自动后台化 —— 把多个 launch 链进同一条命令,
> 只有第一个真的发射出去(真实踩过,靠 subagent 自愈才补救)。

## 先跑 doctor

真实运行前,`python scripts/doctor.py` 确认你 import 的是对的 `dataos`(cwd 里的 `./dataos`
会盖掉安装的包)、核心依赖 + `train` / `eval` extra 在不在、`llamafactory-cli` 在不在 PATH、
GPU 可不可见。它能抓住每次第一跑都会咬人的那些坑。

## 复现(manifest / `--replay`)—— docker 可选

**复现内核不依赖 docker。** 每个实验跑完自动落 `manifest.json`(环境版本 + 数据 revision/fingerprint + config +
seeds + 硬件 + `image@digest`);复现 = 干净重跑 + 验证,**直接 `run_one.py --replay`,无需 docker**:

```bash
python run-dataos/scripts/run_one.py --replay <study>/experiments/<name>/manifest.json --tol 0.3
```

`--replay` 在独立 `_replay/` 里按 manifest 的**声明式数据**(source+revision+recipe)+ train/eval config + seeds 重跑,
验证 **数据 fingerprint 一致** 且 **|Δscore| ≤ --tol** → PASS/FAIL(退出码)。环境一致靠 **`uv.lock`**(`uv sync --frozen`
钉 python 层)。注:用 `set()` 逃生口注入数据(`source=="<injected>"`)的实验不能从 manifest 重跑数据;
**eval-only 实验**(基线,没跑过 data/train)的 replay 自动退化为**只复评**(跳过 fingerprint 比对,只验分数)。

### 可选:docker 环境钉死

docker 是**可选的更强环境钉死**——在 python 层(uv.lock)之外,把 OS / CUDA 整层也冻成 `image@digest`(跨机环境完全一致):

- **镜像**:`run-dataos/assets/Dockerfile`(从 `uv.lock` 构建)。build 后用 `image@sha256:<digest>` 引用(别用 `:tag`,会动)。
- **跑 / 复现**:`run-dataos/scripts/run_in_docker.sh`(薄封装 `docker run … run_one.py`),`-e DATAOS_IMAGE=<digest>` 让 manifest 记下镜像:
  ```bash
  IMAGE=dataos@sha256:<digest> GPU=6 WORKSPACE="$PWD/workspace" \
    run-dataos/scripts/run_in_docker.sh --replay /ws/<study>/experiments/<name>/manifest.json --tol 0
  ```
- **诚实边界**:docker 锁环境,但 GPU 驱动/硬件是宿主的、cuDNN/TF32 非确定 → 跨卡数值仍会漂(manifest 记硬件以
  解释);openai_compatible 评在线服务不可复现;train 要复现记得设 `seed`(见 `train-eval.md`)。

## subagent 自愈并行执行模式

多个实验铺到多卡 + 自动扛失败时用这个模式(**纯运行时编排,DataOS 不感知**):

**角色只有两种**:主 agent 拥有 project 级一切共享状态,sub-agent 只负责**一轮 Experiment 的
data → train → eval**。不设第三种常设角色。

- **主 agent(planner/owner)**:project 开工时一次性做完所有共享态工作——**装好并钉死环境**(dataos
  extras、外部 harness 依赖——独立 venv、钉版本/commit,落 uv.lock / 镜像;**环境只归主 agent**)、
  **评测 discovery→对分校验→冻结**(读目标场景自带的评分器,拿已知模型对出和官方一致的分,钉 commit 填
  `harness` 字段,产出一条所有实验共用的 eval 配置,常为 `suite`;调研可以临时派 subagent,但裁定与落地归
  主 agent)、写 plan、备好共享数据(study 级 prepare-once + `save_to_disk` + 哈希)、**按 GPU 容量扇出**
  (N 张卡 ≤ N 并发,用信号量;别无脑起一堆),最后汇总 `summary()` / `review()`。
- **每个 Experiment 一个 sub-agent**:拿着冻结好的 spec 跑完整的 **data → train → eval**
  (`run_one.py --project … --spec …`,走 docker 则换 `run_in_docker.sh`);**失败诊断重试**
  (OOM→降 batch / 配置错→改 spec / 数据形状错→改 recipe),跑完读回 `report.json` / `manifest.json` 上报。
  它不碰共享状态:不装依赖(缺了上报)、不改评测协议、不动别的实验。
- **铁律:修复只进 spec 或镜像,绝不进 live 容器。** 在运行中的容器里 `pip install` / 手改文件 = 不可复现
  (下次干净 `docker run` 就没了)。配置类改动写回 spec(进 manifest);依赖类改动 = 重 bump 镜像(换 digest)。
- **铁律:subagent 绝不带着飞行中工作终止。** 发射了 nohup 训练/评测就要么等到 terminal、要么把自己发射的
  进程杀干净再退;"我发射了、等跑完会有人分析"是空头承诺(真实事故:R3 发射 3 个 formal 评测后自行退出,
  主 agent 被迫接管)。进度 checkpoint 文件**在收尾时也要更新**——停在"in flight"的进度文件会误导接管者
  (以 report.json 为准,进度文件只是导航)。
- **铁律:持久环境变更只归主 agent。** `pip install` / 改 venv / clone / 动仓库代码 = 共享状态变更——subagent
  **一律不做**(缺依赖就上报,主 agent 装好并落锁后再继续;真实事故:subagent 自行 clone 把整棵树打回旧版)。
  **进程级 env 注入不受限**(`HF_TOKEN=… CUDA_VISIBLE_DEVICES=…` 前缀单条命令,不留痕、不互踩)。
- **discovery → freeze → replay**:subagent 探索/自愈得到结果 + 一份钉死的 manifest;**跑通 ≠ 可复现**——关键实验
  跑完再 `--replay` 干净验证一遍(fingerprint + score),过了才算。
- **交付物**不是每实验一个 bespoke 镜像/脚本,而是:**共享 `image@digest` + 每实验一份 `manifest.json` + 一行 `docker run`**。
