# Troubleshooting — 运维与集群真实坑

这份文件收集 DataOS 在真实运行中踩过的、难以重新获得的运维坑。训练超参/数据层面的坑见
`src/tools/docs/train/training_pitfalls.md`；两次全量批次的系统性复盘见 `docs/known-issues.md`。

## SSH / 远程执行

- **经 ssh 远程 `pkill -f` 会自杀**：`ssh host 'pkill -f "uv pip install"'` 的模式会匹配这条 ssh
  自己的远端 shell 命令行（含同样字样）→ 把自己杀掉，表现为 exit 255、输出消失。用字符类防自匹配：
  `pkill -f "uv pip insta[l]l"`。
- **一条 ssh 只发射一个实验**：并行 N 个实验 = N 条独立 ssh 命令。把多个 launch 用 `&` 链进同一条
  ssh 会被 agent harness 整条后台化，实际只有第一个真的发射。
- **launch 和 poll 分两条 ssh**：launch 命令（nohup &）应立即返回；完成判断用另一条 ssh 轮询，
  别把 `sleep; tail` 挂在 launch 那条上。
- **PATH 必须绝对**：远程训练前 `PATH=<repo>/.venv/bin:$PATH` 要用绝对路径——相对路径在子进程切换
  cwd 后会 FileNotFoundError。
- **ssh agent 认证失败**：偶发 `agent refused operation` 重试即可；连续 `Permission denied
  (publickey)` 先 `ssh-add -l` 看 agent 进程是否还活着（应用退出/锁定后重试无用）。
- **长跑必须 detach**：nohup / tmux，别依赖 ssh 长连接（断线 SIGHUP 会杀训练）。

## HuggingFace 下载

- **hf-mirror 不代理 Xet CDN**（`cas-bridge.xethub.hf.co`）：新模型权重走 Xet 时绕开镜像被墙，
  报成迷惑的 `RuntimeError: ... downloading using hf_transfer`。解法：
  `HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0` 再重拉（或先 `snapshot_download`）。
- **"已缓存"可能只有 config/index、缺权重分片**——真实训练/评测前确认分片齐全。
- **门控（gated）模型/数据**：先在 HF 网页接受条款，token 以 `HF_TOKEN=hf_xxx <命令>` 形式临时
  注入环境。**token 只走 env，绝不写进任何 spec/config/脚本/落盘文件。** 注意 hf-mirror 通常不能
  代理门控仓库（鉴权不透传），门控下载要直连官方端点。

## 集群（H100 Slurm）

- **GPU Guard**：前 30 分钟显存峰值 <24576MiB 的 job 会被守护杀掉（防占卡不用）。LoRA/小模型
  训练峰值可能只有 ~15GiB——sbatch 里先分配并持有一块 ≥16GiB 的常驻 CUDA tensor 把峰值抬过线
  （已写进 `configs/environments/PTB_H100_slurm/env.yaml` 的 discipline，experimenter 开局自带）。
- **libcuda 链接**：部分节点（gpu01/02）只有运行时 `libcuda.so.1`、缺链接符号；触发
  triton/inductor 编译的作业须在 sbatch 里建 job-local 符号链接并 export LIBRARY_PATH /
  TRITON_LIBCUDA_PATH（完整写法见 env.yaml discipline）。
- **不要把本地路径泄进远端命令**：agent 主机的 `/zssd/...` 在集群不存在；远端只用 RUNTIME CONFIG
  给出的集群路径。

## 代码同步（rsync 时代遗留，monorepo + git 部署后基本过时）

- 恢复/对齐远端树用 `rsync -azI`（-I 忽略时间戳全量覆盖）+ 删 `__pycache__`；绝不在 rsync 管理的
  树上 `git clone/checkout/reset`（会把未推送的新代码打回旧版本，留下新旧混合树）。
- 加了新模块后 editable 安装要重装一次（静态 finder）；`scripts/doctor.py` 会比对 import 的
  版本和仓库 pyproject，专抓混合树。
- 当前部署方式是 git checkout（见 AGENTS.md §H100 cluster facts），以上仅在树被手动动过时有用。
