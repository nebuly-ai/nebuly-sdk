##@ General

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Dev

.PHONY: lint
lint:
	poetry run pre-commit install
	poetry run pre-commit run -a

.PHONY: format
format: ## Run the auto-formatter
	@poetry run black .
	@poetry run isort .
	@echo "\033[0;32m[Autoformat OK]\033[0m"

.PHONY: lint-fix
lint-fix: format ## Run the linter and fix issues
	@poetry run ruff check . --fix
	@echo "\033[0;32m[Linting OK]\033[0m"

##@ Test
.PHONY: test
test: ## Run the tests
	@poetry run pytest --cov=nebuly --cov-context=test --cov-report=html --cov-report=xml
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
	@pre-commit install
	@pre-commit install-hooks
