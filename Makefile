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

.PHONY: setup db-up db-down db-schema db-load db-views db-all etl analytics-eda train-yield train-extreme prescriptive-report export-sim embed-sim test-sim dashboards test-dashboards test lint format clean help

help:
	@$(PY) -c "print('targets: setup db-up db-down db-schema db-load db-views db-all etl analytics-eda train-yield train-extreme prescriptive-report export-sim embed-sim test-sim dashboards test-dashboards test lint format clean')"

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

db-schema:
	$(PY) -m database.load --schema

db-load:
	$(PY) -m database.load

db-views:
	$(PY) -m database.load --views

db-all:
	$(PY) -m database.load --all

etl:
	$(PY) -m etl.run --all

analytics-eda:
	$(PY) -m analytics.run_eda

train-yield:
	$(PY) -m analytics.predictive.train_yield

train-extreme:
	$(PY) -m analytics.predictive.train_extreme

prescriptive-report:
	$(PY) -m analytics.prescriptive.report

export-sim:
	$(PY) -m simulator.export_sim_data

embed-sim:
	$(PY) -m simulator.embed

test-sim:
	$(PY) scripts/test_sim.py

dashboards:
	$(PY) -m streamlit run dashboards/app.py

test-dashboards:
	$(PY) scripts/screenshot_dashboards.py

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
