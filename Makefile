
.PHONY: all lint typecheck

test-all: lint typecheck test

test:
	./.venv/bin/pytest

lint:
	./.venv/bin/ruff check ./video_similarity_search

typecheck:
	./.venv/bin/mypy ./video_similarity_search

edit-install:
	pip install -e .[dev]

install:
	pip install .