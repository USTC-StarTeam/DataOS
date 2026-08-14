UV ?= uv
PYTHON ?= 3.12

.PHONY: install-dev lint format format-check test typecheck check

install-dev:
	$(UV) sync --python $(PYTHON) --extra dev

lint:
	$(UV) run --extra dev ruff check .

format:
	$(UV) run --extra dev ruff format .

format-check:
	$(UV) run --extra dev ruff format --check .

test:
	$(UV) run --extra dev pytest

typecheck:
	$(UV) run --extra dev mypy

check: lint format-check typecheck test
