# 实现计划（测试先行）

> 设计见 `docs/DESIGN.md`。原则：**测试先写、能跑再加、勿增实体、缺失能力降级不硬造**。
> 首批 provider：`claude` / `kimi` / `qwen`（均已登录、可真机测）；其余留接口（加一份 spec 即可）。

## 阶段 0 — 脚手架
- `pyproject.toml`（包名 `dataos_agent`，pytest，最小依赖：仅 `pydantic`；CLI 走子进程无需各家 SDK）、`.gitignore`。
- `tests/` 目录 + `pytest.ini`/`conftest.py`。
- ✅ 判据：`pytest` 能空跑；`import dataos_agent` 成功。

## 阶段 1 — 核心类型（先测）
- 测：`tests/test_types.py` —— `Result` 构造/`__str__`=text/status 枚举合法性；`Usage.add`；`Event` 四类。
- 实现：`types.py`（Result / Event / Usage / Step / Status / AgentConfig）、`errors.py`。
- ✅ 判据：test_types 绿。

## 阶段 2 — 结构化输出（先测）
- 测：`tests/test_output.py` —— 合法 JSON→校验过对象；非法/缺字段→`OutputParseError`；从"混杂文本里抠 JSON"。
- 实现：`_output.py`（`parse_output(text_or_file, schema) -> obj`）。
- ✅ 判据：test_output 绿。

## 阶段 3 — 通用 CLI runner + fake provider + 门面（先测）
- 测：`tests/test_query_fake.py` / `tests/test_client_fake.py` —— 用 `fake` provider（确定性）验 `query()` 出 Result、`schema=` 出 data、`Client` 流出 4 类 `Event` 并归约、断连→`status=disconnected`。
- 实现：`_cli.py`（拼 argv + 起子进程 + 逐行 stream-json 读取）、`providers/base.py`（ProviderSpec + registry）、`providers/fake.py`、`agent.py`（`query` + `Client` + 门面）、`__init__.py`。
- ✅ 判据：fake 全绿；`query("...", provider="fake")` 一行能跑。

## 阶段 4 — 三个真 provider spec（先测 mapper，再真机 smoke）
- 先用**录制的真实 stream-json 片段**做 fixture（`tests/fixtures/{claude,kimi,qwen}.jsonl`）。
- 测：`tests/test_providers_map.py` —— 每家 fixture → 中立 `Event` + `Result`（离线、确定性）。
- 实现：`providers/{claude,kimi,qwen}.py`（bin + flag map + event mapper）。
- 真机 smoke（需登录，标 `@pytest.mark.integration`）：`tests/integration/test_smoke.py` —— 对三家各跑 `query("回复 OK", tools=False)`，断言 `status=done` 且 text 非空；跑一个带 `schema=` 的结构化用例。
- ✅ 判据：map 测离线全绿；smoke 三家真机通过。

## 阶段 5 — 断连自愈（v0.3.0 落地）
- 自愈下沉（从 DataOS `llm/net_retry.py` 保真移植）：`_retry.py`（网络错误白名单 + 指数退避 ≤30s）+ `agent.py::_healing_stream`（识别三类断连形态 → 退避 → 带抢救到的 `session_id` 重连续跑）。
- `AgentConfig` 新增 `timeout`（墙钟预算，覆盖重连；到期抛 `TimeoutError`）与 `net_retries`（默认 6，`0` 关闭；无 `timeout` 时的硬上界）；`extra.timeout` → `extra.idle_timeout`（breaking）。
- claude mapper：`error_max_turns` → `status="max_steps"`（良性可续跑边界，不再是 error）；runner 兜底：error 文本命中白名单 → `status="disconnected"`。
- 测：`tests/test_retry.py` —— 重连次数/退避/resume 传递/重试耗尽透出/超时 raise/max_turns 映射，全离线（scripted provider）。
- ✅ 判据：全量测试绿；DataOS 侧可删掉自己的 net_retry 直接继承。
- 未做（留后续）：`tests/integration/test_matrix.py` 跨 provider 一致性、`Result.files` 快照、qwen resume、`effort` 映射。

## 依赖与边界
- **仅** `pydantic`（config/schema 校验）；不引入 claude-agent-sdk / 各家 SDK（统一走 CLI 子进程）。
- 缺失能力**降级**（如某家无 system-prompt flag → 前置进 prompt），不硬造、不泄漏。
- 后续 provider = 新增 `providers/<name>.py` 一份 spec + 一份 fixture 测；核心零改动。
