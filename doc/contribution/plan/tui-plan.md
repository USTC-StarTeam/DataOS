# TUI Reconstruction Plan (Agent-Executable)

## Goal
Implement a Python TUI mode for `py_agent` on top of existing runtime/provider/tools/session/MCP/skills.
Target: terminal interactive workflow only. No Web UI, no RPC.

## Scope
- In-scope:
  - `--mode tui` interactive terminal loop
  - Message input/output rendering
  - Streaming-friendly event panel (tool_call/tool_result)
  - Session continue/new session UX
  - MCP tools and skills availability indicator in UI
  - Graceful interrupt (`Ctrl+C`) and exit commands (`/exit`, `/quit`)
- Out-of-scope:
  - Browser UI
  - Remote RPC server mode
  - Multi-user collaboration

## Constraints
- Python 3.11+
- Environment/tooling: `uv`
- Test-first for each task
- Deterministic tests must not rely on external model API
- No regression to existing print mode

## Proposed Architecture
- New module: `py_agent/modes/tui_mode.py`
- Optional UI helpers:
  - `py_agent/tui/view.py` (render primitives)
  - `py_agent/tui/controller.py` (input loop state machine)
- Reuse existing:
  - runtime: `py_agent/runtime.py`
  - provider: `py_agent/providers/*`
  - tools/mcp/skills/session

## CLI Contract
- Add `--mode tui` to args choices.
- Behavior:
  - `--mode text|json` requires `-p/--print` (existing one-shot mode)
  - `--mode tui` ignores `-p`; enters interactive loop
- Optional flag:
  - `--tui-no-color` (disable ANSI)

## Task Breakdown (Test-First)

### T1: Arg/Entry Wiring
- Tests:
  - parse `--mode tui`
  - mode validation for `-p` combinations
- Impl:
  - update `py_agent/args.py`
  - branch in `py_agent/cli.py` to `run_tui_mode(...)`
- Done when:
  - CLI can start TUI path without calling print mode

### T2: Minimal TUI Loop
- Tests:
  - fake stdin/stdout single roundtrip
  - `/exit` terminates with code 0
  - empty input ignored
- Impl:
  - `run_tui_mode(provider, model, cwd, session_store, ...)`
  - REPL loop: read input -> run runtime -> render output
- Done when:
  - interactive loop functions with non-tool responses

### T3: Event Rendering + Tool Visibility
- Tests:
  - tool_call and tool_result are rendered in deterministic text format
  - long content truncation policy tested
- Impl:
  - render event timeline from runtime events
  - concise status lines for tools
- Done when:
  - user can inspect tool invocation flow in TUI

### T4: Session UX in TUI
- Tests:
  - start with `--continue` loads history summary
  - new turns append to session file
- Impl:
  - display prior-turn count and latest summary at startup
  - persist each completed turn
- Done when:
  - session behavior in TUI matches print mode semantics

### T5: Skills/MCP Awareness
- Tests:
  - TUI startup shows discovered skills count
  - TUI startup shows enabled MCP servers/tools count
- Impl:
  - reuse skills discovery and MCP registry creation from CLI path
  - expose status line in header
- Done when:
  - TUI clearly shows active capability surface

### T6: Error/Interrupt Handling
- Tests:
  - provider error surfaced without crashing loop
  - `KeyboardInterrupt` handled cleanly
- Impl:
  - structured error rendering and continue-to-next-input behavior
- Done when:
  - robust operator experience under failure

## Test Matrix
- Unit:
  - `tests/test_args_tui.py`
  - `tests/test_tui_mode.py`
  - `tests/test_tui_render.py`
- Integration (local deterministic):
  - fake provider + fake tools REPL simulation
- Non-blocking task-eval:
  - optional real-model interactive script smoke

## CI Policy
- Add TUI unit tests to blocking `unit` job.
- Keep real-model tests out of blocking path.
- No new secret requirement for deterministic TUI tests.

## Acceptance Criteria
- `uv run pytest -m "not integration and not task_eval and not mcp_external"` passes
- `uv run python -m py_agent.main --mode tui` launches REPL
- TUI supports:
  - prompt input
  - assistant response rendering
  - tool event rendering
  - `/exit`
  - session continue semantics
- Existing print mode behavior unchanged

## Rollout Order
1. T1 + T2
2. T3 + T4
3. T5 + T6
4. README update + CLI examples + finalize
