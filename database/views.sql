-- Analytical views for ClimateResilience-India. Re-runnable: drops and recreates both views.
-- Column meanings and leakage-safety are documented in database/README.md.
--
-- Season definition (same as config/settings.yaml): kharif = Jun 1 .. Oct 31 of crop year Y,
-- rabi = Nov 1 of Y .. Mar 31 of Y+1. "Sowing" is taken as the first day of the season.
-- Forecast lead L (30/60/90): the forecast is issued L days after season start, so only weather
-- from day 0 .. L-1 and satellite composites that had *finished* (date + 15 days) by then are used.

DROP MATERIALIZED VIEW IF EXISTS v_training_table;
DROP MATERIALIZED VIEW IF EXISTS v_season_weather;

CREATE MATERIALIZED VIEW v_season_weather AS
WITH year_range AS (
    SELECT generate_series(min(extract(year FROM date))::int, max(extract(year FROM date))::int) AS year
    FROM weather_daily
),
seasons AS (
    SELECT d.district_id, y.year, s.season,
           make_date(y.year, s.start_month, 1) AS season_start,
           (make_date(y.year + s.span_years, s.end_month, 1) + INTERVAL '1 month' - INTERVAL '1 day')::date AS season_end
    FROM districts d
    CROSS JOIN year_range y
    CROSS JOIN (VALUES ('kharif', 6, 10, 0), ('rabi', 11, 3, 1)) AS s(season, start_month, end_month, span_years)
),
daily AS (
    SELECT s.district_id, s.year, s.season, s.season_start, s.season_end,
           (w.date - s.season_start) AS day_idx,
           w.rain_mm, w.tmax_c, w.tmin_c,
           (w.rain_mm < 2.5) AS is_dry
    FROM seasons s
    JOIN weather_daily w
      ON w.district_id = s.district_id AND w.date BETWEEN s.season_start AND s.season_end
),
-- Gaps-and-islands: consecutive dry days share (day_idx - row_number).
dry_runs AS (
    SELECT district_id, year, season, MIN(day_idx) AS start_idx, MAX(day_idx) AS end_idx
    FROM (
        SELECT district_id, year, season, day_idx,
               day_idx - ROW_NUMBER() OVER (PARTITION BY district_id, year, season ORDER BY day_idx) AS grp
        FROM daily
        WHERE is_dry
    ) islands
    GROUP BY district_id, year, season, grp
),
spells AS (
    SELECT district_id, year, season,
           MAX(end_idx - start_idx + 1) AS longest_dry_spell_days,
           -- Longest spell inside the first L days: clip each run to day L-1.
           COALESCE(MAX(LEAST(end_idx, 29) - start_idx + 1) FILTER (WHERE start_idx <= 29), 0) AS dry_spell_d30,
           COALESCE(MAX(LEAST(end_idx, 59) - start_idx + 1) FILTER (WHERE start_idx <= 59), 0) AS dry_spell_d60,
           COALESCE(MAX(LEAST(end_idx, 89) - start_idx + 1) FILTER (WHERE start_idx <= 89), 0) AS dry_spell_d90
    FROM dry_runs
    GROUP BY district_id, year, season
),
agg AS (
    SELECT district_id, year, season, season_start, season_end,
           COUNT(*) AS n_days,
           SUM(rain_mm) AS total_rain,
           COUNT(*) FILTER (WHERE rain_mm >= 2.5) AS rainy_days,
           AVG(tmax_c) AS mean_tmax,
           COUNT(*) FILTER (WHERE tmax_c >= 40) AS heat_days,
           SUM(GREATEST((tmax_c + tmin_c) / 2.0 - 10.0, 0)) AS growing_degree_days,
           -- season-to-date features at each forecast lead (days 0 .. L-1 only)
           SUM(rain_mm) FILTER (WHERE day_idx < 30) AS rain_d30,
           COUNT(*) FILTER (WHERE rain_mm >= 2.5 AND day_idx < 30) AS rainy_days_d30,
           AVG(tmax_c) FILTER (WHERE day_idx < 30) AS mean_tmax_d30,
           COUNT(*) FILTER (WHERE tmax_c >= 40 AND day_idx < 30) AS heat_days_d30,
           SUM(GREATEST((tmax_c + tmin_c) / 2.0 - 10.0, 0)) FILTER (WHERE day_idx < 30) AS gdd_d30,
           SUM(rain_mm) FILTER (WHERE day_idx < 60) AS rain_d60,
           COUNT(*) FILTER (WHERE rain_mm >= 2.5 AND day_idx < 60) AS rainy_days_d60,
           AVG(tmax_c) FILTER (WHERE day_idx < 60) AS mean_tmax_d60,
           COUNT(*) FILTER (WHERE tmax_c >= 40 AND day_idx < 60) AS heat_days_d60,
           SUM(GREATEST((tmax_c + tmin_c) / 2.0 - 10.0, 0)) FILTER (WHERE day_idx < 60) AS gdd_d60,
           SUM(rain_mm) FILTER (WHERE day_idx < 90) AS rain_d90,
           COUNT(*) FILTER (WHERE rain_mm >= 2.5 AND day_idx < 90) AS rainy_days_d90,
           AVG(tmax_c) FILTER (WHERE day_idx < 90) AS mean_tmax_d90,
           COUNT(*) FILTER (WHERE tmax_c >= 40 AND day_idx < 90) AS heat_days_d90,
           SUM(GREATEST((tmax_c + tmin_c) / 2.0 - 10.0, 0)) FILTER (WHERE day_idx < 90) AS gdd_d90
    FROM daily
    GROUP BY district_id, year, season, season_start, season_end
    HAVING COUNT(*) = (season_end - season_start + 1)      -- complete seasons only
)
SELECT a.district_id, a.year, a.season, a.season_start, a.season_end,
       a.total_rain, a.rainy_days,
       COALESCE(sp.longest_dry_spell_days, 0) AS longest_dry_spell_days,
       a.mean_tmax, a.heat_days, a.growing_degree_days,
       a.rain_d30, a.rainy_days_d30, COALESCE(sp.dry_spell_d30, 0) AS dry_spell_d30,
       a.mean_tmax_d30, a.heat_days_d30, a.gdd_d30,
       a.rain_d60, a.rainy_days_d60, COALESCE(sp.dry_spell_d60, 0) AS dry_spell_d60,
       a.mean_tmax_d60, a.heat_days_d60, a.gdd_d60,
       a.rain_d90, a.rainy_days_d90, COALESCE(sp.dry_spell_d90, 0) AS dry_spell_d90,
       a.mean_tmax_d90, a.heat_days_d90, a.gdd_d90,
       -- latest composite that had finished (date + 15 d) by sowing / sowing + L
       s0.ndvi  AS ndvi_at_sowing, s0.soil_moisture  AS soil_moisture_at_sowing,
       s30.ndvi AS ndvi_d30,       s30.soil_moisture AS soil_moisture_d30,
       s60.ndvi AS ndvi_d60,       s60.soil_moisture AS soil_moisture_d60,
       s90.ndvi AS ndvi_d90,       s90.soil_moisture AS soil_moisture_d90,
       -- full-season (post-hoc) satellite summaries: NOT leakage-safe for lead forecasts
       pk.ndvi_peak, pk.soil_moisture_mean,
       COALESCE(pk.satellite_synthetic, false) AS satellite_is_synthetic
FROM agg a
LEFT JOIN spells sp ON sp.district_id = a.district_id AND sp.year = a.year AND sp.season = a.season
LEFT JOIN LATERAL (
    SELECT ndvi, soil_moisture FROM satellite_16d s
    WHERE s.district_id = a.district_id AND s.date + 15 <= a.season_start AND s.ndvi IS NOT NULL
    ORDER BY s.date DESC LIMIT 1) s0 ON true
LEFT JOIN LATERAL (
    SELECT ndvi, soil_moisture FROM satellite_16d s
    WHERE s.district_id = a.district_id AND s.date + 15 <= a.season_start + 30 AND s.ndvi IS NOT NULL
    ORDER BY s.date DESC LIMIT 1) s30 ON true
LEFT JOIN LATERAL (
    SELECT ndvi, soil_moisture FROM satellite_16d s
    WHERE s.district_id = a.district_id AND s.date + 15 <= a.season_start + 60 AND s.ndvi IS NOT NULL
    ORDER BY s.date DESC LIMIT 1) s60 ON true
LEFT JOIN LATERAL (
    SELECT ndvi, soil_moisture FROM satellite_16d s
    WHERE s.district_id = a.district_id AND s.date + 15 <= a.season_start + 90 AND s.ndvi IS NOT NULL
    ORDER BY s.date DESC LIMIT 1) s90 ON true
LEFT JOIN LATERAL (
    SELECT MAX(ndvi) AS ndvi_peak, AVG(soil_moisture) AS soil_moisture_mean, bool_or(is_synthetic) AS satellite_synthetic
    FROM satellite_16d s
    WHERE s.district_id = a.district_id AND s.date BETWEEN a.season_start AND a.season_end) pk ON true;

CREATE UNIQUE INDEX v_season_weather_pk ON v_season_weather (district_id, year, season);


-- One row per district-year-season-crop with the yield label and leakage-safe features.
CREATE MATERIALIZED VIEW v_training_table AS
WITH yields AS (
    SELECT c.*,
           CASE WHEN LAG(c.year) OVER w = c.year - 1 THEN LAG(c.yield_t_per_ha) OVER w END AS prev_year_yield,
           -- mean of the previous three crop years that exist (never includes the current year)
           AVG(c.yield_t_per_ha) OVER (PARTITION BY c.district_id, c.season, c.crop
                                       ORDER BY c.year::int RANGE BETWEEN 3 PRECEDING AND 1 PRECEDING) AS yield_mean_prev3,
           COUNT(*) OVER (PARTITION BY c.district_id, c.season, c.crop
                          ORDER BY c.year::int RANGE BETWEEN 3 PRECEDING AND 1 PRECEDING) AS n_prev_years
    FROM crops_annual c
    WINDOW w AS (PARTITION BY c.district_id, c.season, c.crop ORDER BY c.year)
)
SELECT y.district_id, d.state, y.year::int AS year, y.season, y.crop,
       y.yield_t_per_ha AS yield_label,
       y.area_ha, y.production_t, y.yield_flag, y.is_synthetic AS label_is_synthetic,
       y.prev_year_yield, y.yield_mean_prev3, y.n_prev_years,
       sw.season_start, sw.season_end,
       sw.total_rain, sw.rainy_days, sw.longest_dry_spell_days, sw.mean_tmax, sw.heat_days, sw.growing_degree_days,
       sw.rain_d30, sw.rainy_days_d30, sw.dry_spell_d30, sw.mean_tmax_d30, sw.heat_days_d30, sw.gdd_d30,
       sw.rain_d60, sw.rainy_days_d60, sw.dry_spell_d60, sw.mean_tmax_d60, sw.heat_days_d60, sw.gdd_d60,
       sw.rain_d90, sw.rainy_days_d90, sw.dry_spell_d90, sw.mean_tmax_d90, sw.heat_days_d90, sw.gdd_d90,
       sw.ndvi_at_sowing, sw.soil_moisture_at_sowing,
       sw.ndvi_d30, sw.soil_moisture_d30, sw.ndvi_d60, sw.soil_moisture_d60, sw.ndvi_d90, sw.soil_moisture_d90,
       sw.ndvi_peak, sw.soil_moisture_mean, sw.satellite_is_synthetic
FROM yields y
JOIN districts d ON d.district_id = y.district_id
JOIN v_season_weather sw
  ON sw.district_id = y.district_id AND sw.year = y.year AND sw.season = y.season
WHERE y.yield_t_per_ha IS NOT NULL;

CREATE UNIQUE INDEX v_training_table_pk ON v_training_table (district_id, year, season, crop);
CREATE INDEX v_training_table_year_ix ON v_training_table (year);
