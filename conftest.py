# Repo-root conftest: its mere presence makes pytest put the repo root on sys.path,
# so `uv run pytest` (console script, no cwd on sys.path) can resolve cross-test
# imports like `from tests.agent.test_subagent_executor import _cfg`.
