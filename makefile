PYTHON := uv run python3
MYPY_FLAGS := --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

ARGS ?=

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	$(PYTHON) -m src $(ARGS)

debug:
	$(PYTHON) -m pdb -m src $(ARGS)

clean:
	find . -name .venv -prune -o -type d -name "__pycache__" \
		-exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .ruff_cache

lint:
	uv run flake8 src/
	uv run mypy src/ $(MYPY_FLAGS)
