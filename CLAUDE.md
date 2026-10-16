# CLAUDE.md - ClimateResilience-India

Read this at the start of every session. It is the shared memory that keeps conventions from drifting
between phases. If you change an interface or a data contract, update this file in the same commit.

## Project

Predictive analytics for extreme weather and crop-yield optimization in India.

Pipeline: ingest -> PostgreSQL/PostGIS warehouse -> descriptive/diagnostic analytics -> yield and
extreme-weather ML -> LP irrigation optimization -> Three.js 3D simulator -> Streamlit dashboards ->
SMS/WhatsApp advisories.

- **Granularity: DISTRICT.** Village-level yield data is not publicly available (see docs/DECISIONS.md D-001).
  Features are gridded (0.25 degree) weather and satellite data aggregated over district polygons.
- **Crops:** rice, wheat, cotton, sugarcane. **Seasons:** kharif (Jun-Oct), rabi (Nov-Mar).
- **Development region:** Tamil Nadu, Andhra Pradesh, Telangana, Karnataka. Configurable in `config/settings.yaml`.
- **Targets in the original idea (RMSE 2.9, R2 0.82, F1 0.75, 30% water saving) are hypotheses, not results.**
  Report only what our own evaluation produces.

## Environment and commands

- Python 3.11 virtualenv in `.venv`. Repo root is the working directory; run modules as `python -m pkg.module`.
- `make` targets: `setup db-up db-down db-all etl analytics-eda train-yield train-extreme prescriptive-report export-sim embed-sim pipeline dashboards site bot test test-sim test-dashboards test-bot lint format clean`.
  On this Windows machine GNU make is absent: use `mingw32-make` (same targets).
- Database: Docker `postgis/postgis:16-3.4`, container `climateresilience-db`, host port from `DB_PORT`
  (default **5434**, because 5432/5433 are taken locally). Connection string is `DB_URL` in `.env`.
- Config: `from config import load_settings, get_path, set_global_seed, year_range`.
- Before declaring a phase done: `make test` and `make lint` must be green, then commit and tag `phase-N-done`.

## Phase map and interfaces

| Phase | Scope | Key interface / output |
|---|---|---|
| 0 | Scaffold, CLAUDE.md, Docker | `config.load_settings()`, `make test` |
| 1 | ETL | `python -m etl.run --all` -> `data/processed/{districts,weather_daily,satellite_16d,crops_annual}.parquet`, `docs/DATA_QUALITY.md` |
| 2 | Warehouse | `python -m database.load`; views `v_season_weather`, `v_training_table` (materialized); `database/README.md` |
| 3 | Descriptive + diagnostic analytics | `make analytics-eda`; figures in `reports/figures`; `docs/FINDINGS.md` (generated); `analytics/common.py` |
| 4 | Yield models | `analytics.predictive.predict.predict_yield(district_id, crop, season, overrides=None)` -> `{"yield_t_ha","lo","hi","model"}`; `predict_batch`; `models/yield_*`; `reports/yield_model_comparison.md` |
| 5 | Extreme weather early warning | `risk_forecast(district_id, as_of_date)` -> `{"heatwave_p","drought_p","level","lead_days"}` plus `drought_relevant`, `hazard`, `heatwave_source`, `heatwave_p3/p7` (D-024); `forecast_date_range()`; `models/heatwave_lstm.pt`, `models/drought_xgb.joblib` |
| 6 | Prescriptive optimization | `analytics/prescriptive/{water_balance,irrigation_lp,crop_switch}.py`; CLI `python -m analytics.prescriptive.run`; `config/crop_params.yaml`, `config/economics.yaml` |
| 7 | 3D simulator | `simulator/export_sim_data.py` -> `simulator/sim_data.json` (schema v2, `sim_districts.json`); `simulator/terrain.py` (real relief tiles); `simulator/index.html`; `python -m simulator.embed` -> `index_embedded.html`; paddy rice is a ponded field (D-029) |
| 8 | Streamlit dashboards | `make dashboards` (= `streamlit run dashboards/app.py`); router `app.py`, pages in `dashboards/pages/`, shared code in `dashboards/components/`; numbers read from `reports/*.json`; colour tokens in `dashboards/components/theme.py` (D-032); `make test-dashboards` runs the browser check |
| 9 | Farmer advisory | `advisory.generate.generate(district_id, crop, as_of_date, lang) -> str` / `compose(...)`; `advisory/twilio_client.py` (`send_sms`, `send_whatsapp`, dry run by default); `advisory/bot.py` (FastAPI `/whatsapp`, `/health`); `scripts/bulk_advisory.py`; `make bot`, `make test-bot`; `advisory/README.md` |
| 10 | Hardening, deploy, report | `make pipeline`; `Dockerfile`, compose `app`/`bot`; `requirements.txt` (serving) and `requirements.lock`; `python -m config.bundle make|restore|verify` (demo bundle `data/sample`); `scripts/make_report.py` -> `reports/final_report.{md,pdf}`, README results block, `docs/DEMO_SCRIPT.md`, `docs/CV_BULLETS.md`; `scripts/check_container.py`; `docs/DEPLOY.md`; `scripts/check_licenses.md`; static site for Vercel: `python -m scripts.make_site [--dashboards-url URL] [--design field|almanac]` -> `web/`, `vercel.json` (D-033) |

Dependencies flow downward only: a phase may import from earlier phases, never later ones. If a later
phase needs something that does not exist yet (e.g. the dashboards need `advisory.generate`), create a
clearly documented, working implementation of the agreed interface, not a stub that raises.

## Conventions

- Type hints on all public functions; docstrings (what, args, returns) on modules and public functions.
- `ruff check` and `ruff format --check` clean (`make lint`). Line length 100.
- `pathlib.Path` for every path; never hard-code absolute paths. Resolve via `config.get_path(...)`.
- `logging` (module-level `LOG = logging.getLogger(__name__)`), never `print` outside CLI entry-point summaries.
- Seeded randomness everywhere (`config.set_global_seed()`, `numpy.random.default_rng(seed)`); seed 42.
- Complete files only. Never leave TODO, pseudocode, ellipses or "implement later".
- Idempotent CLIs: skip existing outputs unless `--force`.
- Tests must not need network or credentials (use tiny fixtures in `tests/fixtures/`); DB tests auto-skip
  when the database is unreachable.
- Large/regenerable outputs (`data/`, `models/`) are git-ignored; `.env` is never committed.
- Commits and pull requests carry the owner as sole author: no `Co-Authored-By` trailers and no tool or assistant attribution (owner preference; it would show up as a GitHub contributor).

## HARD RULES

1. **Never fabricate real-world data or metrics.** If a source is unreachable or lacks credentials, write a
   clearly named synthetic generator under `etl/synthetic/` and mark every output row `is_synthetic=True`.
   Real data always takes precedence over synthetic data when present.
2. Every number in docs, reports, README and CV bullets must be computed from result files (`reports/*.json`
   or the database). If a metric is missing, print a visible `MISSING` marker; do not invent a value.
3. If the training data is synthetic, the first line of every report that uses it says so.
4. Never leave TODO or pseudocode; always deliver complete, runnable files.
5. Advice to farmers is template-driven and deterministic (no free-form LLM text); sending is dry-run by
   default and needs explicit flags plus env vars; translations need native-speaker review before real use.
6. Report negative results plainly (e.g. if the best model barely beats the naive baseline).

## Data contracts (processed parquet = warehouse tables)

All tables except `districts` carry `is_synthetic: bool` (True for any row produced by a synthetic generator).
`district_id` is a stable lowercase slug `"<state>__<district>"` built from normalised names
(`etl/names.py`), e.g. `tamil_nadu__trichy`. **A "district" is a yield-reporting unit** (ICRISAT 1966 parent
district reconstructed from 2021 polygons, see docs/DECISIONS.md D-006), 51 units in the 4 study states.
**Windows (settings.yaml):** crop years 2000-2017 (real yields end 2017; history from 1991 for lags),
weather 1991-2024, satellite 2000-2024. **Crop year Y** = the season starting in calendar year Y.
Current data status: weather real (IMD + NASA POWER), crops real (ICRISAT), **satellite SYNTHETIC until GEE is authenticated**.

### districts
| column | dtype | notes |
|---|---|---|
| district_id | str | primary key |
| state | str | canonical state name |
| district | str | canonical district name |
| centroid_lat | float64 | EPSG:4326 |
| centroid_lon | float64 | EPSG:4326 |
| geometry_wkt | str | WKT MultiPolygon, EPSG:4326 |
| area_km2 | float64 | computed in an equal-area CRS (EPSG:6933) |
| member_districts | str | extra: `\|`-joined modern (2021) district names inside the unit |
| icrisat_name | str | extra: ICRISAT district name exactly as in the yield table |

### weather_daily  (key: district_id + date)
| column | dtype | notes |
|---|---|---|
| district_id | str | FK districts |
| date | datetime64[ns] | daily |
| rain_mm | float32 | 0-1000 |
| tmax_c | float32 | 0-55 |
| tmin_c | float32 | below tmax |
| rh_pct | float32 | 0-100 |
| wind_ms | float32 | >= 0 |
| solar_mj | float32 | MJ m-2 day-1, >= 0 |
| is_synthetic | bool | |

### satellite_16d  (key: district_id + date)
| column | dtype | notes |
|---|---|---|
| district_id | str | FK districts |
| date | datetime64[ns] | start of the 16-day MODIS composite |
| ndvi | float32 | scaled by 0.0001, range -0.2..1 |
| soil_moisture | float32 | volumetric soil water m3/m3 in 0-1 (ERA5-Land layer 1; SMAP starts only in 2015) |
| is_synthetic | bool | |

### crops_annual  (key: district_id + year + season + crop)
| column | dtype | notes |
|---|---|---|
| district_id | str | FK districts |
| year | int16 | crop year (calendar year in which the season starts) |
| season | str | `kharif` or `rabi` |
| crop | str | rice, wheat, cotton, sugarcane |
| area_ha | float64 | |
| production_t | float64 | |
| yield_t_per_ha | float64 | recomputed as production/area (sugarcane is probably jaggery-equivalent, D-007) |
| yield_flag | bool | reported yield differs from recomputed by more than 5 percent |
| is_synthetic | bool | |

## Evaluation rules

- **Time-based splits only.** Never random-split a time series. Yield: train <= 2018, validation 2019-2021,
  test 2022-2024 **in the original plan; adapted to train <= 2011, validation 2012-2014, test 2015-2017 because
  real yields end in 2017 (D-006); always read the split from settings.yaml**. Extreme weather: train <= 2014,
  validation 2015-2019, test 2020-2024. Tune hyperparameters
  on validation years only; touch the test years once.
- Report RMSE, MAE, R2 (and MAPE for yield) for **every model and a naive baseline** (last-year yield;
  climatology and persistence for extreme events). Rare events: PR-AUC, F1, Brier, reliability diagram.
- Features must be **leakage-safe**: only information available at the forecast date. Add tests that assert it.
- Also report leave-one-state-out CV for spatial generalisation, and 80 percent prediction intervals with
  empirical coverage.
- Fixed seeds must reproduce metrics within 1 percent.
- State the limitations honestly (district aggregation, ecological fallacy, spatial autocorrelation, synthetic data).

## Working loop per phase

Plan -> implement -> run tests and acceptance commands -> fix -> append surprises to `docs/DECISIONS.md` ->
commit -> tag `phase-N-done`. Review your own diff as a skeptical reviewer (leakage, date off-by-ones, units, seeds).
