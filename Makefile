check:
	poetry run pre-commit install
	poetry run pre-commit run -a

test:
	poetry run pytest --cov=nebuly --cov-context=test --cov-report=html --cov-report=xml
