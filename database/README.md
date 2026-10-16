# Warehouse (Phase 2)

PostgreSQL 16 + PostGIS 3.4 in Docker (`make db-up`; host port from `DB_PORT`, default 5434).

```bash
make db-all      # schema -> load parquet -> create views
make db-schema   # schema.sql only
make db-load     # TRUNCATE + COPY all four tables in one transaction, asserts row counts, refreshes views
make db-views    # (re)create v_season_weather and v_training_table
```

| file | purpose |
|---|---|
| `schema.sql` | tables, keys, CHECK constraints (physical ranges), GiST indexes |
| `load.py` | idempotent loader (`python -m database.load [--schema\|--views\|--all]`) |
| `views.sql` | `v_season_weather`, `v_training_table` (both MATERIALIZED) |
| `sample_queries.sql` | eight analytical queries (variability, trends, drought, 100 km neighbours, ...) |

`weather_daily` is not partitioned: it has ~0.6 M rows, far below the 5 M-row threshold in the plan
(see docs/DECISIONS.md D-011). The primary key `(district_id, date)` is the btree the plan asks for.

## Season and lead conventions

* kharif = Jun 1 - Oct 31 of crop year Y; rabi = Nov 1 of Y - Mar 31 of Y+1. "Sowing" = first day of the season.
* A forecast issued at lead **L** (30, 60 or 90 days) may only use weather from season day 0 to L-1 and satellite
  composites that had *finished* by then. A MODIS composite dated `d` covers `d .. d+15`, so a composite is usable
  at date `t` only if `d + 15 <= t`. Columns suffixed `_d30/_d60/_d90` follow this rule; `*_at_sowing` uses `t = season_start`.
* Only complete seasons (every day present) appear in `v_season_weather`.

## `v_season_weather` (one row per district, crop year, season)

| column | meaning | leakage-safe for a forecast at lead L? |
|---|---|---|
| `total_rain`, `rainy_days` (>= 2.5 mm), `longest_dry_spell_days` (< 2.5 mm runs), `mean_tmax`, `heat_days` (Tmax >= 40 C), `growing_degree_days` (base 10 C, daily mean) | whole-season aggregates | **No** - use only for analysis (Phase 3) or post-season nowcasts |
| `rain_dL`, `rainy_days_dL`, `dry_spell_dL`, `mean_tmax_dL`, `heat_days_dL`, `gdd_dL` (L = 30, 60, 90) | same statistics over season days 0..L-1 | **Yes** |
| `ndvi_at_sowing`, `soil_moisture_at_sowing` | latest composite finished by season start | **Yes** |
| `ndvi_dL`, `soil_moisture_dL` | latest composite finished by season_start + L | **Yes** |
| `ndvi_peak`, `soil_moisture_mean` | max NDVI / mean soil moisture over the *whole* season | **No** (post-hoc) |
| `satellite_is_synthetic` | any satellite row in the season is synthetic | - (provenance flag) |

## `v_training_table` (one row per district, crop year, season, crop)

| column | meaning | leakage-safe? |
|---|---|---|
| `yield_label` | `yield_t_per_ha` (recomputed production / area) | the label |
| `area_ha`, `production_t`, `yield_flag`, `label_is_synthetic` | metadata about the label | **No** - area/production of the same season are known only after the season; never use as features |
| `prev_year_yield` | yield of the same district/season/crop in year Y-1 (NULL if that year is missing) | **Yes** - previous crop year was harvested before season Y starts |
| `yield_mean_prev3`, `n_prev_years` | mean of the (up to) three previous years' yields and how many exist | **Yes** - never includes year Y |
| all `season_*` / `_dL` / `*_at_sowing` columns | copied from `v_season_weather` | see above |

Satellite columns are NULL before 2000 (MODIS starts in 2000). If `satellite_is_synthetic` is true the NDVI /
soil-moisture columns are synthetic and derived from rainfall: do not interpret them as independent evidence.

## Running the sample queries

```bash
docker compose exec -T db psql -U postgres -f - < database/sample_queries.sql
```
