# Known issues — 2026-07-29 ptb-gsm8k-c6verify run 复盘

来源：ptb-gsm8k-c6verify（29 轮，3h，baseline 41.0 → best 86.0，dataos-train 首验 run）
的完整轨迹分析。按优先级排列，状态见每条末尾。

## P1 收敛后 no-op 空转烧预算（✅ 关闭：非缺陷，2026-07-30）

leader 在 round 12 判定收敛后输出了 17 轮 no-op marker。复盘结论：**这不是要修的问题**——
leader 就不该被代码叫停；那轮它判定"没的可做"只是因为 stale recon 让候选池看起来只有
24 源（见 P3），v2 的 106 源下探索空间真实存在。保持 leader 持续探索的设计，不加收敛检测。

## P2 deterministic bench 的分数不 exact（未修）

`strategy.yaml` 声称 deterministic bench 分数 EXACT，但同一权重在 gsm8k 反复评出
85/86/85/86——导致 leader 花 3 轮做确认性复评。疑似 vLLM 温度非 0 或 batching 非确定。
修法：lm_eval bridge 钉 temp=0 + 固定 seed；做不到就把"±1 噪声带"写进 strategy。

## P3 recon 缓存不随池/bench 失效（✅ 由 fresh-instance 语义根治，2026-07-30）

recon.json（07-19，v1 池 24 源画像）在池子已切到 v2（106 源）后仍被 resume 复用——
**leader 全程对着过期 10 天的菜单规划**。根因是旧语义"同名 workspace = 静默续跑"。
run JSON 重构后每次执行都是全新实例（被占顺延 (1)(2)），recon 必然现跑现画像，
触发面消除。残留边缘（resume 时改了 config 的 fixed_pool）属误操作而非 bug，
不加机制。

## P4 profiler 信号只有格式维度，主题相关源隐身（✅ 已修，2026-07-30）

`target_fit.target_ratio` 只量输出格式匹配：openmathinstruct2（真·数学源）被判 0.0，
leader 的候选菜单里数学源只剩 numinamath 一个。修复（两步）：
① `quality`/`usage_hint` 标签删除（判断归 agent）；② **target_fit 整个删除**——格式契合
本质是模糊判断，正则只会假装精确。新 schema：profiler 只出硬统计（n_rows/n_sampled/
schema/domain{label,scores 全采样证据}/sample/output_format/negatives），**investigator
拿着 eval 契约为每个源写 1-2 句文本 note**（内容是什么、本 bench 需转换什么），leader
按 统计+note 规划。domain 也从"一行定标签"改为全采样证据占比。

## P5 budget 语义跨 invocation 混乱（未修）

budget 是 per-invocation wall-clock，但 round 9-10 的 plan 显示 leader 把它理解成整个
run 的预算（"RUN OVER — 0.009h"），新 invocation 给 3h 后又恢复真实验。修法：context
里标注 per-invocation 语义，或同时给累计已用。

## P6 pool coverage 是 gsm8k 天花板的主因（观察）

v1 池数学源仅 3 个且 2 个格式不匹配；+44 已接近该池上限。换源/混源空间取决于 v2
（106 源）重画像后的构成。待 P3 修复后重评。

## P7 monitor 的 agent console 在 kimi provider 下失明（✅ 已修，2026-07-30）

~~monitor 只读 Claude Code 的 transcript。~~ 改为 **tee 方案**：`dataos_agent` 新增
`AgentConfig.transcript`——每个中立 Event（text/tool_use/tool_result/done）边流边追加到
JSONL 文件（provider 中立，含断连重连全程）。接线：leader →
`_agent/<run>/leader/transcript.jsonl`，experimenter → `<exp_dir>/agent.jsonl`，
investigator → `<project>/investigator.jsonl`（全部落在运行目录，可归档）。
monitor 优先读 tee 文件，旧 claude run 走原 transcript 兼容路径。

## 已解决（本 run 顺带验证/修复）

- ~~deploy 检出 stale + sub-agent 历史篡改~~：H100 deploy 停在旧 master 且有
  judge workers=1 / gsm8k 数据源篡改，已重置到 main（2026-07-29）。
- dataos-train 接入：本 run 两个训练实验全程 launcher 路径，零 fallback。

# Known issues — 2026-08-04 sol56med 批次复盘

来源：6 bench × 10h 全量批（codex CLI / gpt-5.6-sol / effort medium，workspace
`ptb-<bench>-sol56med(1)(1)`）。~52 实验 16 败，**失败 100% 在 codex CLI/传输层，
0 次训练真失败、0 次 eval 真失败**。按用户指示登记，逐项修复。

## P8 CLI 挂死/断流烧预算（部分修复，watchdog 未做）

三种形态：① 64KB 行限断流（`LimitOverrun`，~7 次）——**已修**，dataos-agent
`d1eb7dd` 把 `create_subprocess_exec` limit 提到 16MiB（DataOS `575c05d` 接入）；
② codex 内部卡死（`models_manager` 刷模型列表超时循环 / 卡在交互式 stdin，3 次）——
**未修**；③ 静默挂死（ssh 抖动后模型流永久静默、零事件零报错，3 次）——**未修**。
放大器：`subagent_executor._run_one` 把剩余全部预算作为单次 `agent.run()` 的
timeout，dataos-agent 无 inactivity 检测，monitor 心跳停滞无告警——一次挂死烧掉
整场（healthbench 9.77h 整场报废、bfcl r7 5.6h、humaneval r12 3.3h；合计
~23h/60h）。修法：experimenter 级 inactivity watchdog（transcript N 分钟无事件→杀
+resume，session_id 可续）+ 单实验 timeout 与全局预算解耦 + monitor 停滞告警。

## P9 已完成的分没入账（未修）

report.json 写在流程最后一步；断流发生在"eval 出分（远端 summary.json 已落盘）"和
"回写 report"之间时，本地记 failed/score=None，分白跑。loop 无 salvage 路径（从不
回 run_root 查 summary.json）。实锤 ≥6 例：gpqa r1 18.08 / r4 28.125（=best 水平）/
r3 训完的 12k 模型从未 eval；bfcl r1/r5/r7 三个 2.0；arena r10 生成 250 条只差
judge。二次伤害：leader 把传输 bug 误判为"该方向操作不可行"（bfcl xLAM 三连断流后
review 判死刑）。修法：失败时主动 ssh 捞 run_root 的 `eval/*/summary.json` 入账；
或 report 改增量式（每 stage 完成即落盘）。

## P10 环境坑重复交学费（未修）

每个 experimenter 是全新 session，环境硬事实不沉淀，各轮 report 的
`compatibility_fixes` 雷同。实锤：gpqa 被集群 GPU Guard（前 30min 显存峰值 <24GiB
即杀，LoRA 峰值 ~15GiB 踩线）杀 4 个 job ~2h，16GiB 预留 workaround 隔轮失传重踩；
frozen sft_env 缺 `dataos_operator` 每轮重新 vendor；gsm8k r10 一轮踩 5 坑
（PartitionTimeLimit / save_to_disk 格式 / libcuda 探测 set -e / eval label 必须含
Qwen3）；/zssd 主机路径泄漏到集群。修法：环境事实进 kb（cluster_discipline 加 GPU
Guard 条款、venv 缺包清单、dataos-train 输入格式、eval label 规则），由
runtime_compose 注入 ENV OPS；备选 experimenter run 级 notes 汇总注入。

## P11 dataos-agent 不支持配置 proxy（未修）

provider 的网络代理目前硬编码在 dataos-agent 内部（codex provider 的 `env()` 内置
固定代理地址），用户无法按 run 配置。需求：dataos-agent 暴露 proxy 配置项（如
`AgentConfig.proxy` 或 provider 级 env 覆盖），DataOS run config 可声明——例如给
codex 设 `http://192.168.1.99:8401`。修法：dataos-agent 加可选 proxy 字段（透传
HTTPS_PROXY/HTTP_PROXY 到 CLI 子进程环境），DataOS run_config 加 `agent.proxy` 透传。
