-- ClimateResilience-India warehouse schema (PostgreSQL 16 + PostGIS 3.4).
-- Idempotent: safe to run repeatedly. Views live in views.sql and are dropped/recreated there.
-- Units and meaning of every column are in CLAUDE.md ("Data contracts").

CREATE EXTENSION IF NOT EXISTS postgis;

-- District = yield-reporting unit (ICRISAT 1966 parent district reconstructed from 2021 polygons).
CREATE TABLE IF NOT EXISTS districts (
    district_id      text PRIMARY KEY,
    state            text NOT NULL,
    district         text NOT NULL,
    centroid         geometry(Point, 4326) NOT NULL,
    geom             geometry(MultiPolygon, 4326) NOT NULL,
    area_km2         double precision NOT NULL CHECK (area_km2 > 0),
    member_districts text,
    icrisat_name     text,
    CONSTRAINT districts_geom_valid CHECK (ST_IsValid(geom))
);
CREATE INDEX IF NOT EXISTS districts_geom_gix ON districts USING GIST (geom);
CREATE INDEX IF NOT EXISTS districts_centroid_gix ON districts USING GIST (centroid);
-- Geography index so ST_DWithin(geom::geography, ..., metres) uses an index.
CREATE INDEX IF NOT EXISTS districts_geog_gix ON districts USING GIST ((geom::geography));
CREATE INDEX IF NOT EXISTS districts_state_ix ON districts (state);

-- Daily weather. 633k rows for 51 districts x 34 years, far below the 5M-row threshold at which
-- partitioning by year pays off, so the table is not partitioned (see docs/DECISIONS.md D-011).
CREATE TABLE IF NOT EXISTS weather_daily (
    district_id  text NOT NULL REFERENCES districts (district_id) ON DELETE CASCADE,
    date         date NOT NULL,
    rain_mm      real CHECK (rain_mm BETWEEN 0 AND 1000),
    tmax_c       real CHECK (tmax_c BETWEEN 0 AND 55),
    tmin_c       real CHECK (tmin_c BETWEEN -10 AND 45),
    rh_pct       real CHECK (rh_pct BETWEEN 0 AND 100),
    wind_ms      real CHECK (wind_ms >= 0),
    solar_mj     real CHECK (solar_mj >= 0),
    is_synthetic boolean NOT NULL DEFAULT false,
    PRIMARY KEY (district_id, date)
);
-- The primary key already is the btree on (district_id, date); a date-first index serves
-- "all districts on day X" and date-range scans.
CREATE INDEX IF NOT EXISTS weather_daily_date_ix ON weather_daily (date);

CREATE TABLE IF NOT EXISTS satellite_16d (
    district_id   text NOT NULL REFERENCES districts (district_id) ON DELETE CASCADE,
    date          date NOT NULL,
    ndvi          real CHECK (ndvi BETWEEN -0.2 AND 1),
    soil_moisture real CHECK (soil_moisture BETWEEN 0 AND 1),
    is_synthetic  boolean NOT NULL DEFAULT false,
    PRIMARY KEY (district_id, date)
);

-- Key is (district_id, year, season, crop): the plan's "all five" key columns do not exist, these are the four.
CREATE TABLE IF NOT EXISTS crops_annual (
    district_id    text NOT NULL REFERENCES districts (district_id) ON DELETE CASCADE,
    year           smallint NOT NULL CHECK (year BETWEEN 1950 AND 2100),
    season         text NOT NULL CHECK (season IN ('kharif', 'rabi')),
    crop           text NOT NULL,
    area_ha        double precision NOT NULL CHECK (area_ha > 0),
    production_t   double precision NOT NULL CHECK (production_t > 0),
    yield_t_per_ha double precision NOT NULL CHECK (yield_t_per_ha > 0),
    yield_flag     boolean NOT NULL DEFAULT false,
    is_synthetic   boolean NOT NULL DEFAULT false,
    PRIMARY KEY (district_id, year, season, crop)
);
CREATE INDEX IF NOT EXISTS crops_annual_crop_year_ix ON crops_annual (crop, year);
