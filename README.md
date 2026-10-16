# ClimateResilience-India

Predictive analytics for extreme weather and crop-yield optimization in India, at **district level**.

Pipeline: ingest -> PostgreSQL/PostGIS warehouse -> descriptive/diagnostic analytics -> yield and
extreme-weather ML -> LP irrigation optimization -> Three.js 3D simulator -> Streamlit dashboards ->
SMS/WhatsApp advisories.

Status: under construction, built phase by phase (see `CLAUDE.md` for the phase map and conventions).

```bash
make setup      # venv + dependencies + .env
make db-up      # PostgreSQL + PostGIS in Docker
make test       # run the test suite
```

On Windows without GNU make, use `mingw32-make` (same targets).
