install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

PHONY: test
test:
	PYTHONPATH=. pytest

format:
	black .

all: install test format
