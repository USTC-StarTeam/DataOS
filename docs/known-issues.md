# Known issues — 2026-07-29 ptb-gsm8k-c6verify run 复盘

来源：ptb-gsm8k-c6verify（29 轮，3h，baseline 41.0 → best 86.0，dataos-train 首验 run）
的完整轨迹分析。按优先级排列，状态见每条末尾。

## P1 收敛后 no-op 空转烧预算（未修）

leader 在 round 12 正确判定收敛后，因 schema 禁止 stop decision，只能每轮编
"terminal-submit-marker" no-op plan——**29 轮中 17 轮（59%）是空转**（~1.5h 预算 +
十几轮 agent 调用）。修法（择一）：(a) loop 检测连续 ≥2 个"data 空+train 空+parent==best"
的 plan → 直接 best_over_tree 收官；(b) schema 允许 STOP 决策。

## P2 deterministic bench 的分数不 exact（未修）

`strategy.yaml` 声称 deterministic bench 分数 EXACT，但同一权重在 gsm8k 反复评出
85/86/85/86——导致 leader 花 3 轮做确认性复评。疑似 vLLM 温度非 0 或 batching 非确定。
修法：lm_eval bridge 钉 temp=0 + 固定 seed；做不到就把"±1 噪声带"写进 strategy。

## P3 recon 缓存不随池/bench 失效（未修，本次实证）

recon.json（07-19，v1 池 24 源画像）在池子已切到 v2（106 源）后仍被 resume 复用——
**leader 全程对着过期 10 天的菜单规划**。修法：recon.json 记录
`{fixed_pool, bench, task}` 指纹，`_load_recon` 不符即判 stale 重跑。

## P4 profiler 信号只有格式维度，主题相关源隐身（未修）

`target_fit.target_ratio` 只量输出格式匹配：openmathinstruct2（真·数学源）被判 0.0，
leader 的候选菜单里数学源只剩 numinamath 一个。修法：profiler 增加主题/域相关性维度
（bench kind → domain 匹配），字段改名 `format_fit_ratio` 防误读；strategy 补一条
"提前收敛且预算有余时，宣布关闭前试一次源多样性探针（混 top-2 主题相关源）"。

## P5 budget 语义跨 invocation 混乱（未修）

budget 是 per-invocation wall-clock，但 round 9-10 的 plan 显示 leader 把它理解成整个
run 的预算（"RUN OVER — 0.009h"），新 invocation 给 3h 后又恢复真实验。修法：context
里标注 per-invocation 语义，或同时给累计已用。

## P6 pool coverage 是 gsm8k 天花板的主因（观察）

v1 池数学源仅 3 个且 2 个格式不匹配；+44 已接近该池上限。换源/混源空间取决于 v2
（106 源）重画像后的构成。待 P3 修复后重评。

## 已解决（本 run 顺带验证/修复）

- ~~deploy 检出 stale + sub-agent 历史篡改~~：H100 deploy 停在旧 master 且有
  judge workers=1 / gsm8k 数据源篡改，已重置到 main（2026-07-29）。
- dataos-train 接入：本 run 两个训练实验全程 launcher 路径，零 fallback。
