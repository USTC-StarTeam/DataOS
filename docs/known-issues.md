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

## P8 CLI 挂死/断流烧预算（✅ 已修，2026-08-05）

三种形态：① 64KB 行限断流（`LimitOverrun`，~7 次）——dataos-agent
`d1eb7dd` 把 `create_subprocess_exec` limit 提到 16MiB（DataOS `575c05d` 接入）；
② codex 内部卡死（`models_manager` 刷模型列表超时循环 / 卡在交互式 stdin，3 次）；
③ 静默挂死（ssh 抖动后模型流永久静默、零事件零报错，3 次）。
放大器：`subagent_executor._run_one` 把剩余全部预算作为单次 `agent.run()` 的
timeout，dataos-agent 无 inactivity 检测，monitor 心跳停滞无告警——一次挂死烧掉
整场（healthbench 9.77h 整场报废、bfcl r7 5.6h、humaneval r12 3.3h；合计
~23h/60h）。
**修复（stall watchdog，三路合一）**：dataos-agent `84211ee` 把 `idle_timeout`
提升为 `AgentConfig` 一级字段，静默超时杀进程后标 `disconnected`（原为 `timeout`，
healing 层不重试）——三种挂死统一走已有传输自愈：backoff + session_id resume，
单次挂死损失从整场降到 ≤idle_timeout+backoff；DataOS 侧 `agent.idle_timeout_s`
（run JSON 可选键，默认 1800s，0 关闭）经 `build_agent` 接到每个角色；
monitor `/api/live` 对活 run 报 `stalled`（全部 transcript 静默 >15min），
前端状态条亮黄牌。总预算 deadline 不变，兜底防无限续命。

## P9 已完成的分没入账（✅ 已修，2026-08-05）

report.json 写在流程最后一步；断流发生在"eval 出分（远端 summary.json 已落盘）"和
"回写 report"之间时，本地记 failed/score=None，分白跑。loop 无 salvage 路径（从不
回 run_root 查 summary.json）。实锤 ≥6 例：gpqa r1 18.08 / r4 28.125（=best 水平）/
r3 训完的 12k 模型从未 eval；bfcl r1/r5/r7 三个 2.0；arena r10 生成 250 条只差
judge。二次伤害：leader 把传输 bug 误判为"该方向操作不可行"（bfcl xLAM 三连断流后
review 判死刑）。
**修复（双管）**：① 契约改**增量写**（kb/report_contract）——summary.json 一出现
立即写 report.json（status+score，无 analysis），REVIEW 之后再重写补 analysis；
"有分无 analysis"合法，"先有 analysis 盘上没有分"是违约。② loop 加
`_salvage_report`——实验无 terminal report 时，先查本地 partial report 的分，再
ssh 查确定性的 `{run_root}/eval/*/summary.json`；捞到即 `status=completed +
salvaged:true + error 注明来源`，分入树（`_report_score` 不看 status）。
注：gpqa r3 那种"模型训完没 eval"不在本修复范围（无分可捞，只能靠 P8 watchdog
减少发生）。

## P10 环境坑重复交学费（✅ 已修，2026-08-05）

每个 experimenter 是全新 session，环境硬事实不沉淀，各轮 report 的
`compatibility_fixes` 雷同。实锤：gpqa 被集群 GPU Guard（前 30min 显存峰值 <24GiB
即杀，LoRA 峰值 ~15GiB 踩线）杀 4 个 job ~2h，16GiB 预留 workaround 隔轮失传重踩；
frozen sft_env 缺 `dataos_operator` 每轮重新 vendor；gsm8k r10 一轮踩 5 坑
（PartitionTimeLimit / save_to_disk 格式 / libcuda 探测 set -e / eval label 必须含
Qwen3）；/zssd 主机路径泄漏到集群。
**修复（两个落点）**：① env.yaml `discipline` +3 条（GPU Guard + ≥16GiB 预留写法、
/zssd 路径隔离、不显式 --time）——渲染进 ENV OPS 硬约束段；② kb 块：`stage_data`
写明 sft_env（dataos_sft 派生 env）已预装 dataos_operator/dataos_train 禁止 vendor
+ converter 要 save_to_disk 格式；`stage_eval` 写明 label 须以模型家族名开头。
vendor 坑由 monorepo 的 dataos_sft 预装顺带根治（见合并提交）。后续维护约定：每次
run 复盘把 report.compatibility_fixes 里重复出现的条目按归属沉淀到这两处。

# Known issues — 2026-08-05 sol56med2 批次复盘

来源：6 bench × 10h（codex/gpt-5.6-sol/medium，P8/P9/P10 修复后首批）。基建验证满分
（传输层事故 0、丢分 0、环境坑 0，对照上批 16 次事故烧 ~23h）；总分 169.75 vs 上批
149.3。以下为策略/工具层新发现。

## P12 full-SFT 升级触发器被 LoRA 碎末涨幅架空（✅ 已修，2026-08-05）

playbook 的升级条件是"LoRA 停滞 ~2 轮转 full-SFT"，但 LoRA 每轮 +2 的碎末涨幅不构成
停滞——触发器永不触发。实锤：gsm8k 沿 20k→30k→45k LoRA 碎涨到 66.11 收官，整条树
**从没试过 full-SFT**，而上批的 71.3 全靠 LoRA 卡 41 后转 full 跳出；arena 同理
（LoRA 三连败后转 full 立即 1.2→7.23，但因触发晚只追到 9.94 < 上批 12.0）。
**修复（按用户决策：去指令化）**：strategy.yaml 整个删除——所有搜索指令（small-first /
exploit-first / 升级触发器 / one-knob 排序）一并移除，不为难 planner；有价值的内容
降级为**纯事实**并入 leader 骨架末尾的 TIPS 块（full-SFT 证据、eval 噪声带、格式
封顶事实、漂移失败模式），leader 自己推导。kb/operator_selection 的 online 段同步
校准（fork 措辞 → dataos-train online-ops + 已桥接清单）。顺带修了 leader 骨架里
"eval is intentionally sampled" 的过时说法（现为全量）。

## P13 在线算子文档与 train 实际桥接集不一致（✅ 已修，2026-08-05）

operator 的 usage 文档把 selector 系（loss_selector 等）写成可在线用（PolicyRuntime
口吻），但 train 的 `dataos_train.online_ops` entry-point 组只桥接了
`loss_reweighter`（+内建 loss_reweight/epoch_subset_select）。healthbench 的
experimenter 想用在线 loss_selector 课程，`probe_online_op('loss_selector')=false`，
如实报败（处理正确）。历史补注：c6verify 那次"能用"也是 experimenter 手工 port 的
一次性 shim（算法副本 + sitecustomize 黑注册），不是正式通道。
**修复**：`dataos_operator.integrations.dataos_train` 新增通用 `_SelectTrainHook`
适配器（observe→after_forward、decide→on_update_boundary，candidates 按行索引占位
即可），entry-point 注册 `loss_selector` + `delta_loss_selector`；
less/nice/near/tsds 和 odm/doremi mixer（需要 eval/query 集或 domain 通道）在
usage guide 和 train 的 online_ops.md 里明确标注"暂仅离线/PolicyRuntime 直驱"。
operator 0.8.0，集群 dataos_sft 已同步，`known_online_op_names()` 实测 5 个可用。

## P14 humaneval 脆区反复探（观察）

humaneval 19 个出分节点 10 个 0.0：best（39.02，5k 严格清洗 LoRA）第 2 轮确立后，
之后 16 轮无改进且大量变体成片坍缩到零——winning 配方周围超参空间对 1.7B base 极脆。
leader 的纪律（读根因、换维度）让它没卡死，但脆区里反复探的性价比很低。观察项：
是否要在 review/策略层引入"连续 N 个 0.0 即判定脆区、回到 best 做保守确认或换大
维度（full-SFT/换源族）"的引导。
