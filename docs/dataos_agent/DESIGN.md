# 产品设计：多-Agent 统一封装（CLI 后端）

> 分支 `feat/cli-backends`，从 0 实现。原则：**测试先行、不过度工程化、如无必要勿增实体**。
>
> **注（2026-08 修订）**：本文写于分支开发期——codex 已实现（`providers/codex.py`）；消费角色现称
> **leader / experimenter / investigator**（文中的 main-agent / sub-agent）；模块已含 `_retry.py`（断连自愈）。

## 0. 一句话

把多个**成品 code-agent CLI**（claude code / kimi / qwen …）封装到**同一套接口**后面——换一家只改 `provider=`。

## 1. 定位

- 后端全是**成品 CLI**（自带 loop + 原生工具 + 权限 + 会话），我们**驱动**它们，不**搭**它们。
- 采用 **Claude Agent SDK 的"驱动成品"心智**（model / instructions / 工具白名单 / workspace / 权限 / 会话），**泛化到 N 个 `provider`**。
- 只补 Claude 缺、我们必须要的一样：**类型化结构化输出**（`data` + schema，思路借 OpenAI `final_output`）。
- 返回**中立归约对象**，不漏任何一家的原生类型。

## 2. 消费者（DataOS 大脑，两个角色）

| 角色 | 工具 | workspace | 输出 |
|---|---|---|---|
| main-agent（规划/复盘）| 关 | 无 | 结构化 `ExperimentPlan`/`ReviewOut` |
| sub-agent（执行）| 开（bash/file）| run_root | `report.json` + 文件 + steps + usage；长跑、断连自愈 |

两角色 = **同一接口、不同配置**。

## 3. 两个入口（都要）

```python
# 入口 1 —— 一次性
result = query(prompt, *, provider, model, ..., schema=None) -> Result

# 入口 2 —— 交互多轮 / 可打断
async with Client(provider, model, ...) as c:
    await c.send(msg)
    async for ev in c.stream():        # 中立事件: text | tool_use | tool_result | done
        ...
    c.result                           # 本轮归约的 Result
    await c.interrupt()
    # 多轮自动保持会话; resume=session_id 恢复
```

## 4. 核心 config（两入口共用，≈9 字段）

```python
@dataclass
class AgentConfig:
    provider: str                       # claude | kimi | qwen  （多-agent 轴）
    model:    str | None = None         # 缺省 = 该 CLI 默认模型
    instructions: str | None = None     # 系统提示（append-system-prompt）
    effort:   str | None = None         # 思考档位（能映射的家才用）
    tools:    bool | list[str] = True   # 原生工具白名单：True 全开 / [..] 指定 / False 全关
    workspace: str | None = None        # cwd / work-dir；产物落这
    max_steps: int | None = None        # 步数上限（max-turns）
    permission: str = "auto"            # auto(自动批准) | ask | deny 之类
    extra:    dict = {}                 # provider 专属逃生口（透传原生 flag）
```

认证：走各 CLI 的**本地登录**（用户已登录 claude/kimi/qwen），无需在包里配 key。

## 5. Result / Event / 状态

```python
Status = Literal["done", "max_steps", "timeout", "disconnected", "error", "refused"]

@dataclass
class Result:
    text:   str = ""                    # 便捷文本；__str__ 返回它
    data:   Any = None                  # 结构化载荷（给了 schema 就是校验过的对象）
    files:  dict[str, str] = {}         # workspace 产物
    status: Status = "done"
    error:  str | None = None
    usage:  Usage = Usage()             # input/output/total tokens (+ cost 可选)
    steps:  list[Step] = []             # 工具调用 transcript（观测，可选）
    session_id: str | None = None       # 续接用

@dataclass
class Event:                            # 唯一的中立流类型（4 类）
    type: Literal["text","tool_use","tool_result","done"]
    text: str = ""; name: str = ""; args: dict | None = None; data: Any = None
```

来源对照：`text`←各家最终文本；`usage(+cost)`←ResultMessage.usage/total_cost_usd；`status`←is_error 富化；`session_id`←会话；`data`←OpenAI final_output 思路（写文件+解析，可移植）。

## 6. 统一 adapter：一套「子进程 + stream-json」机器

三家 headless 接口高度同构（都 `-p <prompt> --output-format stream-json`），所以：

- **一个通用 runner**（`_cli.py`）：按 provider 的 flag map 拼 argv → 起子进程 → 逐行读 stdout 的 JSON 事件 → 交给该 provider 的 mapper。
- **每个 provider = 一份薄 spec**（`providers/<name>.py`）：`bin` + `flag map` + `event mapper`（把该 CLI 的 JSON 事件 → 中立 `Event`，并归约出 `Result`）。

flag map（首批三家）：

| 概念 | claude | kimi | qwen |
|---|---|---|---|
| bin | `claude` | `kimi` | `qwen` |
| prompt | `-p <>` | `--print -p <>` | `-p <>` |
| 流式 | `--output-format stream-json` | 同左 | 同左 |
| model | `--model <>` | `-m <>` | `-m <>` |
| workspace | `--add-dir <>`(+cwd) | `-w <>` | cwd |
| 系统提示 | `--append-system-prompt <>` | (extra/prompt 前置) | (extra) |
| 自动批准 | `--dangerously-skip-permissions` | `--print`(默认自动) | `--yolo` |
| resume | `--resume <id>` | `-S <id>` | (sessions) |

> 不同家的事件 schema 由各自 mapper 归一；缺失能力（如某家没有 system-prompt flag）在 mapper 里降级（前置进 prompt）或走 extra。

## 7. 结构化输出（可移植）

不赌各家原生 schema。约定：**指令里要求 agent 把结果 JSON 写到 workspace 的约定文件 → 读取 + 校验**（`_output.parse_output` / `OutputParseError`）。对任何 CLI 都成立。给了 `schema=` 时 `result.data` 是校验过的对象。

## 8. 横切能力

- 结构化输出（§7）· 会话续接（resume）· **断连自愈**（子进程异常/网络 → 白名单识别 → 退避 + resume 重连，内置默认开启；重试耗尽 → `status=disconnected`）· typed errors。

## 9. 模块（尽量少）

```
src/dataos_agent/
  __init__.py     # 公开面: query, Client, Result, Event, AgentConfig, errors
  types.py        # Result / Event / Usage / Step / Status / AgentConfig
  errors.py       # OutputParseError / ProviderError / DisconnectError / CLINotFound
  _output.py      # parse_output(schema) + 校验
  _cli.py         # 通用子进程 + stream-json 行读取 runner
  providers/
    base.py       # ProviderSpec 接口 + registry
    claude.py kimi.py qwen.py    # 三份薄 spec
    fake.py       # 测试用确定性 provider
  agent.py        # query() + Client + Agent 门面
tests/            # 测试先行（见 PLAN.md）
```

## 10. 非目标（勿增实体）

不做：hooks / `@tool` 进程内工具 / 子代理定义 / 沙箱 / setting_sources / plugins / server tools / 自定义 SessionStore / 上下文-限流遥测。要么进 `extra` 透传，要么不做。ACP 协议后端留作未来选项，v1 用 stream-json 子进程。

## 11. Provider 范围

- **首批（可测，用户已登录）**：`claude` · `kimi` · `qwen`。
- **后续留接口**：codex 等——加一个 `providers/<name>.py` spec 即可，核心不动。
