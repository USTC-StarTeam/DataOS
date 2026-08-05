# dataos_agent

**一套接口，罩住多个成品 code-agent CLI。** 换一家只改 `provider=`。

```python
from dataos_agent import query

print(query("Reply with exactly: OK", provider="claude"))   # 'OK'
print(query("Reply with exactly: OK", provider="kimi"))     # 'OK'  —— 同一行代码
```

> 我们**驱动**成品 coding agent（claude code / kimi / qwen …），不**搭**框架。
> 每个后端自带 loop + 原生工具 + 权限 + 会话；我们把它们归约成一个**中立的 `Result`**。

---

## 为什么

- **成品 > 自搭**：claude code / codex / kimi / qwen 这类 CLI 的真实 agentic-coding 能力，远强于在通用框架上自己拼的 loop。
- **provider 中立**：大脑（或任何调用方）只写一次逻辑，换底层 agent 只改 `provider=`。
- **薄**：后端走 CLI 子进程（`--output-format stream-json`），依赖只有 `pydantic`，不引入各家 SDK。
- **扛断连**：端点中途断流默认自愈——退避后带断连前的 `session_id` 重连续跑（`net_retries`/`timeout` 可调，长跑 agent 的命根子）。
- **勿增实体**：只暴露每家 CLI 都能对应上的公共核心；各家私货走 `extra` 透传或不做。

设计与取舍：见 [`docs/DESIGN.md`](docs/DESIGN.md)。

## 安装

```bash
pip install -e .          # 依赖仅 pydantic
```
前提：目标 CLI 已装且**已登录**（`claude` / `kimi` / `qwen`）——认证走各 CLI 自己的本地登录，本包不碰 key。

## 两个入口

**一次性 `query`**（同步）
```python
r = query("总结这个仓库", provider="claude", workspace="./repo", tools=True)
print(r.status, r.text, r.usage.cost_usd, r.session_id)
```

**交互多轮 `Client`**（异步，可流式、可打断、自动续接会话）
```python
import asyncio
from dataos_agent import Client

async def main():
    async with Client(provider="kimi", workspace="./ws") as c:
        async for ev in c.stream("列出当前目录"):   # text | tool_use | tool_result | done
            print(ev.type, ev.text or ev.name)
        await c.run("再压缩成一句话")               # 续接同一会话
asyncio.run(main())
```

## 结构化输出（可移植）

```python
from pydantic import BaseModel
class Plan(BaseModel):
    name: str; steps: int

r = query('Output ONLY JSON: {"name":"exp","steps":3}', provider="claude", schema=Plan)
r.data          # Plan(name='exp', steps=3)   —— 不合规抛 OutputParseError
```
不赌各家原生 schema：指示 agent 把结果写成 JSON（文本或 workspace 文件），本包解析 + 校验。

## 配置（`AgentConfig`）

> 完整入参（类型 / 默认 / 各 provider 映射与降级）见 **[docs/API.md](docs/API.md)**。

| 字段 | 说明 |
|---|---|
| `provider` | `claude` \| `kimi` \| `qwen` \| … |
| `model` | 具体模型；`None`＝该 CLI 默认 |
| `instructions` | 系统提示（无对应 flag 的家会前置进 prompt）|
| `effort` | 思考档位（能映射的家才用）|
| `tools` | `False` 纯推理 / `True` 全开 / `["Bash","Read"]` 白名单 |
| `workspace` | cwd；产物落这（→ `result.files`）|
| `max_steps` · `permission` · `extra` | 步数上限 · 权限档位 · provider 逃生口 |

## 返回（`Result`）

`text` · `data`(结构化) · `files` · `status`(`done`\|`max_steps`\|`timeout`\|`disconnected`\|`error`\|`refused`) · `error` · `usage`(input/output/total + cost) · `steps` · `session_id`。`str(result)` = 最终文本。

> 每个返回字段 + `Event` / `Usage` / `Step` / 异常的完整说明见 **[docs/API.md](docs/API.md)**。

## Provider

| provider | 状态 |
|---|---|
| `claude` | ✅ claude code CLI（stream-json）|
| `kimi` | ✅ kimi CLI（session 从 stderr 提取；不报 token usage）|
| `qwen` | ✅ qwen code CLI（结构同 claude，共用同一 mapper）|
| `codex` | ✅ codex CLI（`exec --json`；thread_id 即 resume；usage 全；OpenAI 走备用代理）|
| `fake` | 测试用确定性 provider |

**加一个新 provider**（如 codex）：新增 `src/dataos_agent/providers/<name>.py`——一个 `CliSpec` 子类（`bin` + `build_argv` + `parse_line`，可选 `finalize`）并 `register()`。核心零改动。

## 布局

```
src/dataos_agent/
  types.py errors.py _output.py _cli.py agent.py
  providers/  base.py fake.py claude.py qwen.py kimi.py
tests/        单元(离线) + integration(真机 smoke)
docs/         DESIGN.md USAGE.md
PLAN.md
```

## 开发 / 测试

```bash
pytest -q -m "not integration"      # 离线单元测试（快、确定性）
pytest -m integration -s            # 真机 smoke（需登录 claude/kimi/qwen）
```
测试先行；离线用录制的 stream-json fixture 回放各家 mapper。

## 状态

v1：两入口 + 三 provider + 结构化输出 + 统一 Result，均有测试。详见 [`docs/USAGE.md`](docs/USAGE.md)。
