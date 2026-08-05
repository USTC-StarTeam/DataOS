# API 参考

对 `dataos_agent` 每个入参 / 返回字段的精确说明。速览见 [`USAGE.md`](USAGE.md)，设计见 [`DESIGN.md`](DESIGN.md)。

---

## 1. 入口

### `query(prompt, *, provider, schema=None, **config) -> Result`
同步、一次性。内部起子进程跑到结束，把消息流归约成一个 `Result` 返回。

### `aquery(prompt, *, provider, schema=None, **config) -> Result`
`query` 的异步版，参数完全相同（`await aquery(...)`）。

### `class Client(*, provider, schema=None, **config)`
异步、交互多轮。

| 成员 | 签名 | 说明 |
|---|---|---|
| `async with Client(...) as c` | 上下文管理器 | 退出时自动 `interrupt()` |
| `c.stream(prompt)` | `-> AsyncIterator[Event]` | 逐个产出本轮中立事件；`done` 时写 `c.result` / `c.session_id` |
| `c.run(prompt)` | `-> Result` | 跑完一轮（不流式），返回该轮 `Result` |
| `c.interrupt()` | `-> None` | 尽力关闭在途流（销毁子进程）|
| `c.result` | `Result \| None` | 最近一轮的结果 |
| `c.session_id` | `str \| None` | 当前会话；下一轮自动 `resume` 续接 |

**位置参数**：`prompt: str` —— 发给 agent 的一段话（或指令）。

---

## 2. 入参：`config`（= `AgentConfig` 字段，query/aquery/Client 的 `**config`）

> `AgentConfig` 用 pydantic 校验；**传了未知字段会直接报错**（`extra='forbid'`）。provider-专属的原生 flag 走 `extra`。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `provider` | `str` | **必填** | 后端：`"claude"` / `"kimi"` / `"qwen"` / `"fake"` / 已注册的自定义名 |
| `model` | `str \| None` | `None` | 具体模型名；`None` = 用该 CLI 的默认模型 |
| `instructions` | `str \| None` | `None` | 系统提示 / 前置指令（见 §6 各家映射；无对应 flag 的家会前置进 prompt）|
| `effort` | `str \| None` | `None` | 思考档位（claude `--effort`、codex `-c model_reasoning_effort=<>`；未映射的家可经 `extra.args` 手动传）|
| `tools` | `bool \| list[str]` | `True` | 原生工具策略：`True` 全开自动批准 / `["Bash","Read"]` 白名单 / `False` 纯推理（不启用工具）|
| `workspace` | `str \| None` | `None` | 子进程 `cwd`；agent 产物落这里（→ `Result.files`）|
| `max_steps` | `int \| None` | `None` | 步数上限（映射 `--max-turns`；目前仅 claude 支持）|
| `permission` | `str` | `"auto"` | `"auto"` = 自动批准工具调用；其它值 = 不自动批准（工具需交互/被拒）|
| `timeout` | `float \| None` | `None` | **整次调用的墙钟预算（秒），覆盖传输重连**；到期抛 `TimeoutError`（不可重试）|
| `net_retries` | `int` | `6` | 传输自愈的重连上限（`0` = 关闭自愈）；仅在未设 `timeout` 时作硬上界（设了则由预算约束）|
| `extra` | `dict` | `{}` | 逃生口，见下 |

### 传输自愈（self-healing）

端点在高峰期会中途断连，CLI 本身不会重连。所有入口（`query`/`aquery`/`Client`）默认开启自愈：
识别三类断连形态（流抛异常 / result 事件渲染成 error / 流无结果终止），命中网络错误白名单
（connection closed/reset、broken pipe、429/5xx、overloaded…）后，按指数退避（2s 起、≤30s）
**带断连前抢救到的 `session_id` 重连续跑**。重试耗尽后不再重试，以 `status="disconnected"`
的 Result 透出（不抛异常）；`timeout` 到期抛 `TimeoutError`。白名单之外的错误永不重试。

### `extra` 里被识别的键

| 键 | 类型 | 作用 |
|---|---|---|
| `resume` | `str` | 恢复某会话 id（`Client` 多轮时自动填；自愈重连时自动用断连前的 id；也可手动）|
| `args` | `list[str]` | 原样附加到 CLI 命令末尾的**原生 flag**（透传各家私有参数）|

`idle_timeout` 已提升为 `AgentConfig` 一级字段（不再是 extra 键）：静默看门狗——stdout
N 秒无任何事件 → 杀子进程并标 `status="disconnected"`，healing 层据此 backoff + resume
重连续跑（静默 = 传输故障，不是终态超时）。

其余 `extra.*` 键会被忽略（留给 provider 自定义读取）。

### `schema` 参数（query/aquery/Client 都有）

| | |
|---|---|
| 类型 | `type[pydantic.BaseModel] \| None` |
| 默认 | `None` |
| 行为 | 给了且 `status=="done"` 时，从最终文本抠 JSON 并按 schema 校验 → 填入 `Result.data`（校验过的实例）|
| 失败 | 无合法 JSON / 校验不过 → 抛 `OutputParseError` |
| 容错 | 支持裸 JSON、```json 围栏块、混在 thinking/散文里的 JSON、或 agent 写到 workspace 的 JSON 文件 |

---

## 3. 返回：`Result`

`str(result)` == `result.text`（`print(result)` 直接出最终文本）。

| 字段 | 类型 | 说明 |
|---|---|---|
| `text` | `str` | 最终答案文本（权威值；来自各家的 result 事件）|
| `data` | `Any` | 结构化载荷：给了 `schema=` 就是校验过的对象，否则 `None` |
| `files` | `dict[str, str]` | workspace 里 agent 产生/改动的文件 `{相对路径: 内容}`（v1 暂为空 dict，接口预留）|
| `status` | `Status` | 终止原因，见 §5 |
| `error` | `str \| None` | 失败时的错误消息（`status` 非 `done` 时）|
| `usage` | `Usage` | token 与成本，见 §4 |
| `steps` | `list[Step]` | 工具调用轨迹（观测用）|
| `session_id` | `str \| None` | 本次会话 id（用于续接）|

---

## 4. `Usage` / `Step`

**`Usage`**

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `input_tokens` | `int` | 0 | 输入 token |
| `output_tokens` | `int` | 0 | 输出 token |
| `total_tokens` | `int` | 0 | 合计（未显式给时 = input+output）|
| `cost_usd` | `float` | 0.0 | 成本（美元；仅 claude 直接上报，其它家可能为 0）|

> 注：**kimi** 的 stream-json 不上报 usage，其 `Usage` 全为 0。

**`Step`**（一次工具调用）

| 字段 | 类型 | 说明 |
|---|---|---|
| `tool` | `str` | 工具名（如 `Bash`）|
| `args` | `dict` | 调用参数 |
| `result` | `str` | 工具返回（能拿到时；claude/kimi 会回填，qwen 可能为空）|

---

## 5. `Status`（`Literal`）

| 值 | 含义 | 触发 |
|---|---|---|
| `"done"` | 正常结束 | 后端正常返回（stop_reason=end_turn 等）|
| `"max_steps"` | 撞步数上限 | stop_reason=max_turns/max_tokens |
| `"timeout"` | 空闲超时 | `extra.timeout` 到点 |
| `"disconnected"` | 传输中断 | 子进程崩/流断（**可重试**）|
| `"error"` | 真错误 | 非零退出且流未解释 / result.is_error |
| `"refused"` | 被拒 | 预留（如权限拒绝）|

---

## 6. 流事件：`Event`（`Client.stream` 产出）

| 字段 | 类型 | 何时有值 |
|---|---|---|
| `type` | `"text" \| "tool_use" \| "tool_result" \| "done"` | 总有 |
| `text` | `str` | `type=="text"`：助手文本块 |
| `name` | `str` | `type in (tool_use, tool_result)`：工具名 |
| `args` | `dict \| None` | `type=="tool_use"`：工具参数 |
| `data` | `Any` | `type=="done"`：本轮的 `Result`；`type=="tool_result"`：工具返回文本 |

`stream` 的最后一个事件总是 `type="done"`，`ev.data` 即归约好的 `Result`。

---

## 7. 异常（都继承 `DataOSAgentError`）

| 异常 | 何时抛 |
|---|---|
| `UnknownProvider` | `provider` 名未注册 |
| `CLINotFound` | 后端 CLI 二进制不在 PATH |
| `OutputParseError` | 给了 `schema=` 但输出里没有合法的对应 JSON |
| `ProviderError` | 后端返回可识别的错误（预留；v1 多以 `status="error"` 表达）|
| `DisconnectError` | 传输中断（预留；以 `status="disconnected"` 表达，默认由内置自愈重连，见 §2）|
| `DataOSAgentError` | 上述的基类，一网打尽 |

> `error`/`disconnected` 主要通过 `Result.status` 表达（不抛异常）；`disconnected` 默认已被内置自愈重连，透出即重试耗尽。`CLINotFound`/`UnknownProvider`/`OutputParseError`/`TimeoutError` 会抛。

---

## 8. config → 各 provider 的 CLI 映射（含降级）

| config | claude | kimi | qwen | codex |
|---|---|---|---|---|
| `prompt` | `-p <>` | `--print -p <>` | `-p <>` | 位置参数（`exec` 末尾）|
| （流式）| `--output-format stream-json --verbose` | `--print --output-format stream-json` | `--output-format stream-json` | `exec --json`（JSONL 事件流）|
| `model` | `--model <>` | `-m <>` | `-m <>` | `-m <>` |
| `effort` | `--effort <>` | —（v1 未映射）| —（v1 未映射）| `-c model_reasoning_effort=<>` |
| `instructions` | `--append-system-prompt <>` | **前置进 prompt**（无 flag）| **前置进 prompt**（无 flag）| **前置进 prompt**（无 flag）|
| `tools=list` | `--allowed-tools "<空格分隔>"` | —（前置约定）| —| —（v1 未映射）|
| `tools` 真 + `permission=auto` | `--dangerously-skip-permissions` | `--print`（默认自动批准）| `--yolo` | `--dangerously-bypass-approvals-and-sandbox` |
| `tools=False` | 不加 skip → 工具跑不了（纯推理）| 降级（print 恒自动批准）| 不加 yolo | `-s read-only` |
| `workspace` | `cwd`（+可 `--add-dir`）| `-w <>` | `cwd` | `-C <>`（resume 子命令不接受 `-C`，续接沿用原 cwd）|
| `max_steps` | `--max-turns <>` | —（v1 未映射）| —（v1 未映射）| —（v1 未映射）|
| `extra.resume` | `--resume <id>` | `-S <id>` | —（v1 未映射）| 子命令 `exec resume <thread_id>` |
| `extra.args` | 原样附加 | 原样附加 | 原样附加 | 原样附加 |
| session 来源 | result 事件 | **stderr**（`kimi -r <id>`）| result 事件 | `thread.started` 的 `thread_id` |
| usage 来源 | result 事件（含 cost）| **无**（全 0）| result 事件 | `turn.completed`（无 cost）|

> 缺 flag 的能力**降级**（如 instructions 前置进 prompt），不硬造、不泄漏各家私有形状。需要某家独有 flag 时用 `extra.args` 透传。
