# PR1 Compatibility Baseline (CLI Skeleton)

This baseline defines the minimum observable behavior for PR1.

1. Running `py-agent --help` exits with code `0`.
2. `--help` output contains `usage:`.
3. `--help` output contains `--version` option.
4. Running `py-agent --version` exits with code `0`.
5. `--version` outputs exactly one semantic version line.
6. Running with no args exits with code `0`.
7. Running with no args produces no stderr output.
8. Unknown flag exits non-zero.
9. Unknown flag prints argparse error message.
10. CLI executable entrypoint is `py-agent`.
11. Module entrypoint `python -m py_agent.main --version` works.
12. `pyproject.toml` declares Python `>=3.11`.
13. `uv run pytest` can discover tests from `tests/`.
14. `uv run ruff check .` passes for PR1 code.
15. CI runs on push to `tjh_MVP` and on PR to `main`.
