# dataos_agent 使用指南

一套接口罩住多个**成品 code-agent CLI**（claude code / kimi / qwen …）。换一家只改 `provider=`。
设计见 [`DESIGN.md`](DESIGN.md)。

## 前提
- 目标 CLI 已装且**已登录**：`claude` / `kimi` / `qwen`（认证走各 CLI 自己的本地登录，本包不管 key）。
- `pip install -e .`（依赖仅 `pydantic`；后端走 CLI 子进程，无需各家 SDK）。

## 一次性：`query`
```python
from dataos_agent import query

r = query("Reply with exactly: OK", provider="claude", tools=False)
print(r)                 # 'OK'（__str__ = 最终文本）
print(r.status, r.usage.total_tokens, r.usage.cost_usd, r.session_id)
```

`query` 参数 = `AgentConfig` 字段：`provider` `model` `instructions` `effort`
`tools(bool|list)` `workspace` `max_steps` `permission` `extra`，外加 `schema=`。

## 结构化输出：`schema=`
```python
from pydantic import BaseModel
class Plan(BaseModel):
    name: str; steps: int

r = query('Output ONLY JSON: {"name":"exp","steps":3}', provider="claude", schema=Plan)
assert isinstance(r.data, Plan) and r.data.steps == 3
```
可移植：指示 agent 把结果写成 JSON（文本或 workspace 文件），本包解析+校验，不合规抛 `OutputParseError`。

## 交互 / 多轮：`Client`
```python
from dataos_agent import Client
import asyncio

async def main():
    async with Client(provider="kimi", workspace="./ws") as c:
        async for ev in c.stream("列出当前目录并说明"):
            print(ev.type, ev.text or ev.name)     # text | tool_use | tool_result | done
        print(c.result.status, c.result.files)
        await c.run("再压缩成一句话")               # 自动续接同一会话(resume)
asyncio.run(main())
```

## 工具策略 `tools`
- `tools=False` → 纯推理（不启用原生工具；给 main-agent）
- `tools=True` → 全开、自动批准（给 sub-agent）
- `tools=["Bash","Read"]` → 白名单

## 返回 `Result`
`text` · `data`(结构化) · `files`(workspace 产物) · `status`(done|max_steps|timeout|disconnected|error|refused) · `error` · `usage`(input/output/total + cost) · `steps` · `session_id`。

## 加一个新 provider
新增 `src/dataos_agent/providers/<name>.py`：一个 `CliSpec` 子类（`bin` + `build_argv` + `parse_line`，可选 `finalize`）并 `register()`；在 `providers/__init__.py` 的列表里加名字。核心零改动。

## 已知
- **claude / qwen** 共用同一套 Claude-Code stream-json mapper；三家真机 smoke 均通过（含结构化输出）。
- **kimi** 的 stream-json 不报 token usage（`usage` 为 0），session_id 从 stderr 提取。
