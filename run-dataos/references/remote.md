# 远程执行 —— 本机 Claude Code 驱动远程服务器上的 DataOS

正式场景:**Claude Code 在用户本机,DataOS 在远程 GPU 服务器。** 本机只有这个 skill + SSH;dataos 包、
`train` / `eval` extra、数据、GPU 都在服务器上。你(Claude)通过 ssh 在服务器上执行所有 dataos 调用,把
产物读回本机。

> 服务器是用户的 —— SSH host / 别名由用户给你(例如 `my-gpu`,或 `ssh user@host -p port`)。下面用
> `<host>` 占位,仓库路径用 `<repo>` 占位(例如 `/data/<user>/dev/DataOS`)。

## 一、在服务器上把环境装好(一次)

服务器上要有:dataos 包 + extra + 数据。最省心是 clone 仓库(顺带带上 skill 的脚本):

```bash
ssh <host> '
  git clone <DataOS repo URL> <repo> && cd <repo> &&
  uv sync --extra train --extra eval        # 真实训练+评测;只 stub 则 uv sync
'
```

> 没有 uv 就用项目 venv:`.venv/bin/python -m pip install -e ".[train,eval]"`,或按服务器约定。
> 算子依赖 `dataos-operator`(git);服务器若被墙,挂代理 + 用 PyPI / HF 镜像(见第五节)。

## 二、自检(每次开工先做)

```bash
ssh <host> 'cd <repo> && .venv/bin/python run-dataos/scripts/doctor.py'
```

绿了(或至少"stub 就绪")再往下。它报:dataos 树对不对、核心依赖 / `train`/`eval` extra、
`llamafactory-cli` 在不在 PATH、GPU。

## 三、跑实验(经 ssh)

数据准备 + study 都在服务器上发生。两种粒度:

**A. 计划 / 小事 / 复盘 —— inline `python -c`:**
```bash
ssh <host> 'cd <repo> && .venv/bin/python -c "
import dataos
s = dataos.project(id=\"qwen-gsm8k\").study(id=\"lr-sweep\")
s.set_plan({ ... })
"'
```

**B. 单实验(真实训练,钉一张卡)—— 用仓库里的 run_one.py:**

先在服务器上用算子把数据集准备好并 `save_to_disk`(见 `data-prep.md`),例如存到 `<repo>/prepared/`。
真实训练慢(几分钟~几十分钟),**用 nohup / tmux 后台跑,别让 ssh 连接挂着**(ssh 断了会 SIGHUP 杀掉训练):

```bash
ssh <host> 'cd <repo> &&
  PATH=<repo>/.venv/bin:$PATH HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=6 \
  nohup .venv/bin/python run-dataos/scripts/run_one.py \
    --project qwen-gsm8k --study lr-sweep --spec lr-1e-4 --data prepared/ > /tmp/exp_lr-1e-4.log 2>&1 &'
```
> **PATH 用绝对路径**(`<repo>/.venv/bin`),别用相对 `.venv/bin`(下面坑里解释)。
> **launch 和 poll 分两条 ssh**:上面这条只 launch(nohup &,马上返回);完成判断用**另外**的 ssh until-loop
> 轮询 `report.json`,别把 `sleep; tail` 挂在 launch 那条上(会被 agent harness 自动后台化、还占着连接)。

**并行**:同一个 study,多个 `run_one.py` 各钉不同 `CUDA_VISIBLE_DEVICES`,后台并发。`summary()` 只读盘,
容忍还在跑的实验。

## 四、读回产物

产物在服务器的 `<repo>/<workspace>/<study_id>/` 下。读回本机:

```bash
ssh <host> 'cat <repo>/workspace/lr-sweep/review.json'                              # 看复盘
ssh <host> 'cat <repo>/workspace/lr-sweep/experiments/lr-1e-4/report.json'          # 看单实验
rsync -az <host>:<repo>/workspace/lr-sweep/ ./pulled/lr-sweep/                       # 或整份拉回本机
```

**别凭脑补结论 —— 读真实的 report.json / review.json。**

## 五、服务器上的真实坑

- **PATH(必须绝对)**:`llamafactory-cli` 在 `.venv/bin`,非登录 ssh 不在 PATH → 真实训练前
  `PATH=<repo>/.venv/bin:$PATH`。**必须绝对**——相对 `.venv/bin` 会让 DataOS 的 `shutil.which` 返回相对路径,
  等训练子进程换到 `cwd=exp_dir` 时 `FileNotFoundError`。(DataOS 现已对 which 结果做 abspath 兜底,仍建议绝对。)
- **cwd 阴影**:从仓库根直接 `python -c "import dataos"` 时,cwd 里的 `./dataos` 会盖掉安装的包;用
  `.venv/bin/python`,必要时从干净目录跑,或确认 `dataos.__file__` 指向对的树(doctor 会查)。
- **下载**:服务器若被墙,模型 / 数据集走 `HF_ENDPOINT=https://hf-mirror.com`;pip 用国内镜像;git 挂代理。
- **同步代码**:本机改了 dataos / skill 后,`rsync -az --exclude .git --exclude .venv --exclude '/workspace*'
  ./ <host>:<repo>/` 同步过去;若加了**新模块**,再 `ssh <host> 'cd <repo> && uv pip install -e . --no-deps'`
  (uv 的 editable finder 是静态的)。
- **detach**:长跑用 nohup / tmux,别依赖 ssh 长连接。

## 六、本机离线快验(可选)

本机若也装了核心依赖(`datasets` + `dataos_operator`),可本机只跑 **stub**(`train {"mode":"stub"}`、
`eval {"provider":"stub", "provider_config":{"score":...}}`,不碰 GPU / extras),把流水线逻辑
(plan → study → compare → review)过一遍。真实训练 / 评测一律回服务器。
