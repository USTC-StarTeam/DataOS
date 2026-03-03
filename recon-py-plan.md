# Recon Plan for Code Agent: TypeScript -> Python (Minimal Core + MCP + Skills, No TUI)

## Objective
Rebuild the minimal core of `pi-mono/packages/coding-agent` in Python, and include MCP + skill capabilities.
Do NOT implement TUI, Web UI, or RPC.
Repository: `https://github.com/USTC-StarTeam/DataOS.git`

## Hard Constraints
- Scope remains CLI-only.
- MCP and skills must be usable in print/script mode.
- Deliver in runnable increments; each phase must be executable.
- Test-first enforcement: for every task/PR, tests must be added or updated before implementation is considered complete.
- Development branch: `tjh_MVP`.
- Merge policy: all production merges to `main` must be via Pull Request.
- CI must be implemented during development (starting from PR1), not deferred to the end.
- Development and test execution must use `uv` environment management.

## In-Scope
- CLI entrypoint and argument parsing.
- Print mode (`-p/--print`) with `text` and `json` outputs.
- Minimal runtime loop for tool-calling.
- Local core tools: `read`, `write`, `edit`, `bash`.
- MCP integration (client side) for external tool servers.
- Skills system (`SKILL.md` discovery + explicit invocation + prompt injection).
- Session persistence (JSONL): `--no-session`, `--session`, `--continue`.
- One provider integration only: OpenAI-compatible endpoint (tool-calling capable), tested with model `glm-5`.
  - Test configuration must use environment variables:
    - `STARTEAM_BASE_URL`
    - `STAETEAM_API_KEY`

## Out-of-Scope
- TUI interactive mode (`src/modes/interactive/*`).
- Web UI (`packages/web-ui`).
- RPC mode (`src/modes/rpc/*`, `--mode rpc`).
- Full extension/theme parity.

## Source Scope (Read First)
- `pi-mono/packages/coding-agent/src/main.ts`
- `pi-mono/packages/coding-agent/src/cli.ts`
- `pi-mono/packages/coding-agent/src/cli/args.ts`
- `pi-mono/packages/coding-agent/src/modes/print-mode.ts`
- `pi-mono/packages/coding-agent/src/core/agent-session.ts`
- `pi-mono/packages/coding-agent/src/core/session-manager.ts`
- `pi-mono/packages/coding-agent/src/core/tools/*`
- `pi-mono/packages/coding-agent/src/core/skills.ts`
- `pi-mono/packages/coding-agent/src/core/resource-loader.ts`
- `pi-mono/packages/coding-agent/docs/skills.md`

## Target Python Layout
```text
py-agent/
  pyproject.toml
  README.md
  py_agent/
    main.py
    cli.py
    args.py
    config.py
    core/
      agent_session.py
      session_manager.py
      resource_loader.py
      skills.py
      tools/
        read.py
        write.py
        edit.py
        bash.py
    mcp/
      client.py
      registry.py
      adapters.py
    providers/
      base.py
      openai_responses.py
    modes/
      print_mode.py
    storage/
      jsonl_store.py
  tests/
```

## PR Plan

### PR1: Python Skeleton + CLI Args
Tasks:
- Add/adjust tests first for CLI baseline behavior.
- Create Python package skeleton and `pyproject.toml`.
- Add `uv`-based setup workflow (`uv sync --extra dev`).
- Implement `main.py`, `cli.py`, `args.py`.
- Support `--help` and `--version`.
- Capture CLI baseline and create `compat_baseline.md` (>= 15 checks).
- Create initial CI workflow (`.github/workflows/ci.yml`) with uv setup + lint + unit test jobs.
Done when:
- CLI starts and basic flags work without model integration.
- Code is committed on `tjh_MVP` and PR-ready for `main`.
- CI workflow runs successfully on `tjh_MVP` pushes and `main` pull requests.
- Local verification commands run successfully via `uv run`.

### PR2: OpenAI Provider + Print Mode
Tasks:
- Add/adjust tests first for provider behavior, output modes, and env wiring.
- Implement OpenAI provider adapter in `providers/openai_responses.py`.
- Implement `-p/--print` in `modes/print_mode.py`.
- Support `--mode text|json` outputs.
- Wire env vars for integration testing:
  - `STARTEAM_BASE_URL`
  - `STAETEAM_API_KEY`
- Set default test model to `glm-5` (overridable by CLI arg).
- Extend CI with integration test job gated by repository secrets.
Done when:
- CLI can complete non-tool prompts using `glm-5`.
- JSON mode outputs parseable structured events.
- Integration smoke test passes with `STARTEAM_BASE_URL`, `STAETEAM_API_KEY`, and model `glm-5`.
- Code is committed on `tjh_MVP` and PR-ready for `main`.
- CI passes for both unit and integration tracks.

### PR3: Local Tools (read/write/edit/bash)
Tasks:
- Add/adjust tests first for tool behavior, runtime loop behavior, and failure compatibility paths.
- Implement tool abstraction and local tool registry.
- Implement `read`, `write`, `edit`, `bash`.
- Add tool input validation and structured error output.
- Enforce `bash` safety: cwd boundary, timeout, output truncation.
- Add CI coverage for tool tests and safety-path regressions.
Done when:
- End-to-end tool-calling loop works with local tools.
- Code is committed on `tjh_MVP` and PR-ready for `main`.
- CI includes and passes tool-specific tests.

### PR4: Session Persistence
Tasks:
- Add/adjust tests first for session persistence and resume paths.
- Implement JSONL session store and loader.
- Add `--no-session`, `--session`, `--continue`.
- Persist and restore core event/message history.
- Add CI coverage for session persistence/resume scenarios.
Done when:
- Session can be resumed across process restarts.
- Code is committed on `tjh_MVP` and PR-ready for `main`.
- CI includes and passes session tests.

### PR5: Skills
Tasks:
- Add/adjust tests first for skill discovery/parsing/invocation behavior.
- Implement `SKILL.md` discovery and parsing.
- Implement `/skill:name` expansion.
- Inject available skills metadata into system prompt.
- Add CI coverage for skills discovery/invocation tests.
Done when:
- Skills are discoverable and affect runtime behavior.
- Code is committed on `tjh_MVP` and PR-ready for `main`.
- CI includes and passes skills tests.

### PR6: MCP Integration
Tasks:
- Add/adjust tests first for MCP listing/invocation/error handling behavior.
- Implement MCP client and MCP tool registry.
- Merge MCP and local tools into unified dispatch path.
- Add per-run/server config and explicit enablement controls.
- Add CI coverage using MCP mock server integration tests.
Done when:
- At least one MCP server tool can be listed and invoked in print mode.
- Code is committed on `tjh_MVP` and PR-ready for `main`.
- CI includes and passes MCP integration tests.

## Git Workflow
- Remote repository: `https://github.com/USTC-StarTeam/DataOS.git`.
- Active development branch: `tjh_MVP`.
- All implementation commits are made on `tjh_MVP`.
- For merging to mainline, open Pull Requests with:
  - base branch: `main`
  - compare branch: `tjh_MVP` (or a short-lived feature branch created from `tjh_MVP`).
- Recommended PR sequence: PR1 -> PR2 -> PR3 -> PR4 -> PR5 -> PR6.
- Do not merge directly into `main` without PR review.

## Test Requirements
- All test/lint commands are executed with `uv run`.
- Every task must have explicit test coverage mapping:
  - requirement -> test file -> test case.
- No task is complete if the related tests are missing.
- Unit test per local tool.
- Deterministic `edit` tests.
- Skill parsing/discovery/invocation tests.
- MCP mock-server integration tests (list tools + invoke tool + error path).
- Session persistence/resume tests.
- JSON output schema validation test.
- End-to-end CLI smoke tests.

## Safety Requirements
- `bash` enforces cwd boundary, timeout, output truncation, structured non-zero exit.
- `write/edit` block path traversal outside workspace.
- MCP tools require explicit allowlist or explicit server enablement to execute.

## Output Contract Per PR
After each PR, output:
1. files created/changed,
2. tests added,
3. run commands,
4. known gaps.
5. CI updates (jobs added/changed and latest run status).

## Final Completion Criteria
Complete only if all are true:
- Python CLI-only core is operational.
- OpenAI-compatible provider is operational with tool-calling.
- Integration is verified with `STARTEAM_BASE_URL`, `STAETEAM_API_KEY`, and model `glm-5`.
- Skills are operational (discovery + invocation + prompt integration).
- MCP tools are operational (connected + callable + error-handled).
- TUI/Web/RPC are not implemented.
- Core tools + sessions are stable.
- Tests + packaging are in place.
- Compatibility report is published.
