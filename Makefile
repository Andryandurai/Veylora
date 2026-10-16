# ClimateResilience-India task runner.
# Works with GNU make on Linux/macOS and with mingw32-make on Windows.
# Recipes stick to `$(PY) ...` calls so they behave the same under sh and cmd.exe.

ifeq ($(OS),Windows_NT)
  VENV_PY := .venv/Scripts/python.exe
else
  VENV_PY := .venv/bin/python
endif

# Use the venv interpreter when it exists, otherwise whatever `python` is on PATH.
PY := $(if $(wildcard $(VENV_PY)),$(VENV_PY),python)

.PHONY: setup db-up db-down test lint format clean help

help:
	@$(PY) -c "print('targets: setup db-up db-down test lint format clean')"

setup:
	python -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -e ".[dev]"
	$(VENV_PY) scripts/tasks.py ensure-env

db-up:
	$(PY) scripts/tasks.py ensure-env
	docker compose up -d --wait db

db-down:
	docker compose down

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

clean:
	$(PY) scripts/tasks.py clean
