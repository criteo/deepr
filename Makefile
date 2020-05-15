venv-lint-test: ## [Continuous integration] Install in venv and run lint and test
	python3.6 -m venv .env && . .env/bin/activate && make install install-dev lint test && rm -rf .env

install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip setuptools
	python -m pip install -r requirements.txt
	python -m pip install -e .

install-dev: ## [Local development] Install test requirements and pre-commit
	python -m pip install -r test-requirements.txt

lint: ## [Local development] Run mypy, pylint and black
	python -m mypy deepr
	python -m pylint deepr
	python -m black --check -l 120 deepr

test: ## [Local development] Run unit tests.
	python -m pytest -n 4 -v tests/unit

black: ## [Local development] Auto-format python code using black
	python -m black -l 120 .

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
