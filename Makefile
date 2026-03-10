.PHONY: venv install install-dev
.PHONY: sync reset-venv
.PHONY: help format lint typecheck test precommit check
.PHONY: download-dataset ingest-dataset prepare-dataset build-dataset
.PHONY: clean


VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON := $(VENV)/bin/python

SRC := src
SCRIPTS := scripts

help:
	@echo "Development:"
	@echo "  make format           - format code with black and auto-fix with ruff"
	@echo "  make lint             - run ruff checks"
	@echo "  make typecheck        - run mypy on src/"
	@echo "  make test             - run pytest"
	@echo "  make precommit        - run pre-commit on all files"
	@echo "  make check            - run lint + typecheck + test"
	@echo ""
	@echo "Pipeline:"
	@echo "  make download-dataset - download raw UN/LOCODE data"
	@echo "  make ingest-dataset   - ingest raw data into intermediate format"
	@echo "  make prepare-dataset  - prepare final dataset for retrieval and embedding"
	@echo "  make build-dataset    - run full dataset pipeline"
	@echo ""
	@echo "Utility:"
	@echo "  make clean            - remove common Python cache files"

# ---------------------
# Environment
# ---------------------

venv:
	python -m venv $(VENV)

install: venv
	$(PIP) install -r requirements.txt

install-dev: venv
	$(PIP) install -r requirements-dev.txt
	$(VENV)/bin/pre-commit install

sync:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

reset-venv:
	rm -rf $(VENV)
	make install-dev

# ---------------------
# Development
# ---------------------

format:
	ruff check . --fix
	black .

lint:
	ruff check .

typecheck:
	mypy $(SRC)

test:
	pytest

precommit:
	pre-commit run --all-files

check: lint typecheck test

# ---------------------
# Data pipeline
# ---------------------

download-dataset:
	$(PYTHON) -m $(SCRIPTS).download_dataset

ingest-dataset:
	$(PYTHON) -m $(SCRIPTS).ingest_dataset

prepare-dataset:
	$(PYTHON) -m $(SCRIPTS).prepare_dataset

build-dataset: download-dataset ingest-dataset prepare-dataset

# ---------------------
# Utility
# ---------------------

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
