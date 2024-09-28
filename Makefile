.PHONY: lint format package-dev

lint:
	ruff check src tests demo
	ruff format src tests demo --check

format:
	ruff check src tests demo --fix
	ruff format src tests demo

package-dev:
	uv add --editable .
