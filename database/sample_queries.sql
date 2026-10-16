-- Eight analytical queries that matter for the project. Run with:
--   docker compose exec -T db psql -U postgres -f - < database/sample_queries.sql
-- NDVI / soil-moisture based results are only as real as satellite_16d (check is_synthetic!).

-- 1. Rainfall variability: coefficient of variation of kharif season rainfall per district.
SELECT d.district_id, d.state,
       round(avg(sw.total_rain)::numeric, 0)                          AS mean_kharif_rain_mm,
       round((stddev_samp(sw.total_rain) / avg(sw.total_rain))::numeric, 3) AS cv_kharif_rain
FROM v_season_weather sw JOIN districts d USING (district_id)
WHERE sw.season = 'kharif'
GROUP BY d.district_id, d.state
ORDER BY cv_kharif_rain DESC
LIMIT 15;

-- 2. Yield trend by state and crop: OLS slope (t/ha per year) and mean yield, 2000 onwards.
SELECT state, crop, count(*) AS n_obs,
       round(avg(yield_label)::numeric, 3)                              AS mean_yield_t_ha,
       round(regr_slope(yield_label, year)::numeric, 4)                 AS slope_t_ha_per_year,
       round(regr_r2(yield_label, year)::numeric, 3)                    AS r2
FROM v_training_table
WHERE year >= 2000
GROUP BY state, crop
ORDER BY state, crop;

-- 3. Drought frequency: share of seasons with total rain below 75 % of the district's long-term mean.
WITH normals AS (
    SELECT district_id, season, avg(total_rain) AS mean_rain FROM v_season_weather GROUP BY district_id, season
)
SELECT sw.district_id, sw.season,
       count(*) AS seasons,
       count(*) FILTER (WHERE sw.total_rain < 0.75 * n.mean_rain) AS drought_seasons,
       round((count(*) FILTER (WHERE sw.total_rain < 0.75 * n.mean_rain))::numeric / count(*), 3) AS drought_frequency
FROM v_season_weather sw JOIN normals n USING (district_id, season)
GROUP BY sw.district_id, sw.season
ORDER BY drought_frequency DESC, sw.district_id
LIMIT 15;

-- 4. Neighbours within 100 km of a district (geography, uses districts_geog_gix).
SELECT b.district_id, b.state,
       round((ST_Distance(a.geom::geography, b.geom::geography) / 1000)::numeric, 1) AS km_between_borders,
       round((ST_Distance(a.centroid::geography, b.centroid::geography) / 1000)::numeric, 1) AS km_between_centroids
FROM districts a
JOIN districts b ON b.district_id <> a.district_id
 AND ST_DWithin(a.geom::geography, b.geom::geography, 100000)
WHERE a.district_id = 'tamil_nadu__thanjavur'
ORDER BY km_between_borders, km_between_centroids;

-- 5. Dry-spell ranking: mean of the longest dry spell (days with < 2.5 mm) in the kharif season.
SELECT district_id,
       round(avg(longest_dry_spell_days)::numeric, 1) AS mean_longest_dry_spell_days,
       max(longest_dry_spell_days)                    AS worst_dry_spell_days
FROM v_season_weather
WHERE season = 'kharif'
GROUP BY district_id
ORDER BY mean_longest_dry_spell_days DESC
LIMIT 15;

-- 6. Heat-day trend: slope of days with Tmax >= 40 C per year, 1991-2024, per district (kharif + rabi).
SELECT district_id,
       round(avg(heat_days)::numeric, 2)                  AS mean_heat_days_per_season,
       round(regr_slope(heat_days, year)::numeric, 3)     AS heat_days_per_year_trend
FROM v_season_weather
GROUP BY district_id
HAVING max(heat_days) > 0
ORDER BY heat_days_per_year_trend DESC
LIMIT 15;

-- 7. NDVI versus yield by decile of peak-season NDVI (rice). Only meaningful if NDVI is real.
WITH ranked AS (
    SELECT yield_label, ndvi_peak, satellite_is_synthetic,
           ntile(10) OVER (ORDER BY ndvi_peak) AS ndvi_decile
    FROM v_training_table
    WHERE crop = 'rice' AND ndvi_peak IS NOT NULL
)
SELECT ndvi_decile, count(*) AS n,
       round(avg(ndvi_peak)::numeric, 3)   AS mean_ndvi_peak,
       round(avg(yield_label)::numeric, 3) AS mean_yield_t_ha,
       bool_or(satellite_is_synthetic)     AS any_synthetic_ndvi
FROM ranked
GROUP BY ndvi_decile
ORDER BY ndvi_decile;

-- 8. Top-10 vulnerable districts for rice: drought frequency + yield variability + heat exposure.
--    Score = mean of the three percentile ranks (higher = more vulnerable). Decision support only.
WITH normals AS (
    SELECT district_id, avg(total_rain) AS mean_rain FROM v_season_weather WHERE season = 'kharif' GROUP BY district_id
),
drought AS (
    SELECT sw.district_id,
           (count(*) FILTER (WHERE sw.total_rain < 0.75 * n.mean_rain))::numeric / count(*) AS drought_frequency
    FROM v_season_weather sw JOIN normals n USING (district_id)
    WHERE sw.season = 'kharif'
    GROUP BY sw.district_id
),
variability AS (
    SELECT district_id, stddev_samp(yield_label) / avg(yield_label) AS yield_cv, avg(heat_days) AS mean_heat_days
    FROM v_training_table
    WHERE crop = 'rice' AND season = 'kharif' AND year >= 2000
    GROUP BY district_id
    HAVING count(*) >= 10
)
SELECT v.district_id, d.state,
       round(dr.drought_frequency, 3) AS drought_frequency,
       round(v.yield_cv::numeric, 3)  AS rice_yield_cv,
       round(v.mean_heat_days::numeric, 2) AS mean_heat_days,
       round(((percent_rank() OVER (ORDER BY dr.drought_frequency)
             + percent_rank() OVER (ORDER BY v.yield_cv)
             + percent_rank() OVER (ORDER BY v.mean_heat_days)) / 3)::numeric, 3) AS vulnerability_score
FROM variability v
JOIN drought dr USING (district_id)
JOIN districts d USING (district_id)
ORDER BY vulnerability_score DESC
LIMIT 10;
