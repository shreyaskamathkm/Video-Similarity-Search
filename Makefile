
.PHONY: all lint typecheck

all: lint typecheck

lint:
	ruff check ./video_similarity_search

typecheck:
	mypy ./video_similarity_search
