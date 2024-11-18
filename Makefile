##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Dev

.PHONY: ruff
ruff: ## Run ruff
	@poetry run ruff check .
	@echo "\033[0;32m[Ruff OK]\033[0m"

.PHONY: bandit
bandit: ## Run bandit
	@poetry run bandit -r nebuly/
	@echo "\033[0;32m[Bandit OK]\033[0m"

.PHONY: pylint
pylint: ## Run pylint
	@poetry run pylint nebuly tests
	@echo "\033[0;32m[Pylint OK]\033[0m"

.PHONY: black
black: ## Run black
	@poetry run black .
	@echo "\033[0;32m[Black OK]\033[0m"

.PHONY: isort
isort: ## Run isort
	@poetry run isort .
	@echo "\033[0;32m[Isort OK]\033[0m"

.PHONY: mypy
mypy: ## Run mypy
	@poetry run mypy .
	@echo "\033[0;32m[Mypy OK]\033[0m"

.PHONY: lint
lint: ruff mypy pylint ## Run the linter

.PHONY: format
format: black isort ## Run the auto-formatter

.PHONY: check
check: format lint bandit ## Run all the checks

##@ Test
.PHONY: test
test: ## Run the tests
	@poetry run pytest --cov=nebuly --cov-context=test --cov-report=html --cov-report=xml -v
	@poetry run coverage report

##@ Install
.PHONY: poetry
poetry: ## Install poetry
	@which poetry || curl -sSL https://install.python-poetry.org | python3 -

.PHONY: install-dev
install-dev: ## Install CloudSurfer with all the development dependencies
	@which poetry || pip install poetry
	poetry install --no-interaction --no-root --with dev

.PHONY: setup
setup: poetry install-dev ## Setup the dev environment
