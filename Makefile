.PHONY: lint

lint:
	ruff check src tests demo
	ruff format src tests demo --check

format:
	ruff check src tests demo --fix
	ruff format src tests demo
