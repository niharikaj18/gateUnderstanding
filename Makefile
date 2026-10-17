.PHONY: install test lint serve ui index eval
install:
	pip install -e ".[dev,anthropic]"
test:
	pytest -q
lint:
	ruff check src tests
index:
	gitcontext index
serve:
	gitcontext serve
ui:
	gitcontext ui
eval:
	gitcontext eval generate && gitcontext eval run
