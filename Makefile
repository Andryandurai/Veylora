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

.PHONY: setup db-up db-down db-schema db-load db-views db-all etl analytics-eda train-yield train-extreme prescriptive-report export-sim embed-sim test-sim dashboards test-dashboards bot test-bot pipeline test lint format clean help

help:
	@$(PY) -c "print('targets: setup db-up db-down db-schema db-load db-views db-all etl analytics-eda train-yield train-extreme prescriptive-report export-sim embed-sim test-sim dashboards test-dashboards bot test-bot pipeline test lint format clean')"

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

bot:
	$(PY) -m uvicorn advisory.bot:app --port 8000

test-bot:
	$(PY) scripts/smoke_bot.py

# The whole chain from raw sources to the dashboards' inputs. Every step is idempotent (ETL skips existing downloads).
# Needs Docker for the database; the IMD download is slow the first time (hours). See README.md for the offline demo bundle.
pipeline:
	$(MAKE) db-up
	$(MAKE) etl
	$(MAKE) db-all
	$(MAKE) analytics-eda
	$(MAKE) train-yield
	$(MAKE) train-extreme
	$(MAKE) prescriptive-report
	$(MAKE) export-sim
	$(MAKE) embed-sim

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
