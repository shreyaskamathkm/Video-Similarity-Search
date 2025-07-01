
.PHONY: all lint typecheck

test-all: lint typecheck

lint:
	ruff check ./video_similarity_search

typecheck:
	mypy ./video_similarity_search

edit-install:
	pip install -e .

install:
	pip install .