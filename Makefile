install: install-cpu

install-cpu: ## [Local development, CPU] Upgrade pip, install requirements, install package.
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt
	python -m pip install -e ".[cpu]"

install-gpu: ## [Local development, GPU] Upgrade pip, install requirements, install package.
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements-gpu.txt
	python -m pip install -e ".[gpu]"

install-dev: ## [Local development] Install test requirements and pre-commit
	python -m pip install -r requirements-test.txt

lint: ## [Local development] Run mypy, pylint and black
	python -m mypy deepr
	python -m pylint deepr
	python -m black --check -l 120 deepr

black: ## [Local development] Auto-format python code using black
	python -m black -l 120 .

test: ## [Local development] Run unit tests, doctest and notebooks
	python -m pytest -n 4 -v tests/unit
	python -m pytest --doctest-modules -v deepr
	find docs/getting_started/*.ipynb | xargs jupyter nbconvert --to notebook --execute && rm docs/getting_started/*.nbconvert.ipynb

integration: ## [Local development] Run integration tests.
	python -m pytest -v tests/integration

venv-lint-test-integration: ## [Continuous integration] Install in venv and run lint and test
	python3.6 -m venv .env && . .env/bin/activate && make install install-dev lint test integration && rm -rf .env

build-dist: ## [Continuous integration] Build package for pypi
	python3.6 -m venv .env
	. .env/bin/activate && pip install -U pip setuptools wheel
	. .env/bin/activate && python setup.py sdist
	rm -rf .env

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
