.PHONY: install install-dev test lint train evaluate predict clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	ruff check src tests

train:
	python -m income_classifier.cli train --classifier hgb --iterations 50

evaluate:
	python -m income_classifier.cli evaluate

predict:
	python -m income_classifier.cli predict --json '{"age":35,"workclass":"Private","fnlwgt":200000,"education":"Bachelors","education_num":13,"marital_status":"Married-civ-spouse","occupation":"Exec-managerial","relationship":"Husband","race":"White","sex":"Male","capital_gain":5000,"capital_loss":0,"hours_per_week":50,"native_country":"United-States"}'

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
