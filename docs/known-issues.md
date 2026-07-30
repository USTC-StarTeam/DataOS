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

## P4 profiler 信号只有格式维度，主题相关源隐身（部分修复 2026-07-30）

`target_fit.target_ratio` 只量输出格式匹配：openmathinstruct2（真·数学源）被判 0.0，
leader 的候选菜单里数学源只剩 numinamath 一个。**已修的一半**：`quality`/`usage_hint`
标签已删除——profiler 只给原始统计，判断完全归 leader（strategy 同步改为"读数字和
样本自己判断，domain 相关但格式差的源可以配清洗步骤用，不是禁区"）。**未修的一半**：
`target_ratio` 仍只有格式维度，建议增加主题/域相关性维度并改名 `format_fit_ratio`；
strategy 补"提前收敛且预算有余时试一次源多样性探针"。

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
