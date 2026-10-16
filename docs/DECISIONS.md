# Decision log

Append a dated entry whenever a choice, deviation or surprising finding would otherwise be lost.
The final report (Phase 10) is generated from this file plus `reports/*.json`.

## 2026-10-16 - Phase 0

### D-001: District-level granularity, not village-level
Public crop-yield data (ICRISAT District Level Database, data.gov.in Area-Production-Yield) is
published per district and season. Village-level yield is not openly available, so training a
village model would require fabricating labels. The pipeline therefore predicts yield per
district-season-crop, using gridded weather and satellite features aggregated over each district
polygon. Village-level modelling is documented as future work, not simulated.

### D-002: Google Earth Engine for satellite features
MODIS NDVI (MOD13Q1, 250 m, 16-day), CHIRPS rainfall and SMAP soil moisture are all hosted in
Earth Engine. `reduceRegions` computes district statistics server-side, which avoids downloading
terabytes of rasters. The cost is a one-time `earthengine authenticate` and a registered Cloud
project. If that is missing, Phase 1 falls back to clearly flagged synthetic generators
(`is_synthetic=True`) so the rest of the pipeline can still be built and tested.

### D-003: Local environment adaptations (Windows)
- GNU `make` is not installed; `mingw32-make` is. The Makefile uses only `$(PY) ...` calls and a
  small cross-platform helper (`scripts/tasks.py`) so it works with either under sh or cmd.exe.
- Host port 5432 (and 5433) are already used by other services on this machine, so the project
  database is published on host port **5434** (`DB_PORT` in `.env`). Inside Docker it is still 5432.
- Python 3.11.9 is used for the virtualenv (`.venv`), as the plan specifies.

### D-004: Project database is the default `postgres` database
The Phase 0 acceptance command is `docker compose exec db psql -U postgres -c "SELECT PostGIS_Version();"`,
which connects to the default `postgres` database. The postgis image only enables the extension in
`$POSTGRES_DB`, so the project uses `POSTGRES_DB=postgres` (a dedicated `climate` DB made that command fail
in the first attempt). Tables live in the `public` schema of a local, single-purpose container.

### D-005: Resolved dependency versions (2026-10-16)
pandas 3.0 (string dtype and copy-on-write are the default), numpy 2.4, torch 2.14, xgboost 3.2,
scikit-learn 1.9, statsmodels 0.15, geopandas 1.2, SQLAlchemy 2.1, psycopg 3.3, imdlib 0.3.1,
earthengine-api 1.7. Code must be written for pandas 3 semantics (no chained assignment, `str` dtype).
