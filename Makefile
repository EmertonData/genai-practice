lint:
	uv run ruff check --fix

type-check:
	uv run ty check

clean:
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
