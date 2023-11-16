install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

lint:
	pylint ./**/*.py --fail-under=8

PHONY: test
test:
	PYTHONPATH=. pytest

format:
	black *.py

all: install lint test
