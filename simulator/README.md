# 3D crop-climate simulator (Phase 7, scene upgraded in v1.1)

An **explanation tool**: a browser landscape that shows, week by week, how rainfall, temperature and an irrigation
strategy change soil moisture, crop condition and the predicted yield of a district. It visualises the outputs of the
Phase 4-6 models; it is **not a physical crop-growth simulation** and not advice for farm decisions (the same
sentence is shown in the page footer).

## Run it

```bash
python -m simulator.export_sim_data     # ~2 min: evaluates 2 772 scenarios with the Phase 4-6 code -> simulator/sim_data.json
python -m simulator.embed               # inlines the data and Three.js -> simulator/index_embedded.html
```

Then **double-click `simulator/index_embedded.html`**. It needs no server and no internet (Three.js r160.1 is inlined
from `simulator/vendor/`). `index.html` is the same page without embedded data: served over HTTP
(`python -m http.server -d simulator`) it fetches `sim_data.json`; opened from disk it shows a drop zone for the JSON file.
`index.html?district=<id>&crop=<crop>` preselects a district and crop (used by the Streamlit page); `&quality=high|low|auto`
fixes the level of detail.

The first export downloads the elevation tiles for the six districts (about 1 MB) into `data/raw/elevation`; later runs use
the cache. Without network access and without the cache the relief falls back to the procedural proxy and says so.

Controls: district and crop selectors, rainfall multiplier (0.4-1.6), temperature change (-2 to +4 C), irrigation strategy
(rainfed / farmer baseline / optimized), week scrubber with play/pause and speed (the scrubber moves in days: the date and
the weather of the day are shown), time of day (06:00-19:00), level of detail, compare mode (baseline vs optimized, cameras
synchronised), uncertainty view, screenshot, colour-blind-safe soil palette. Keys: **Space** play, **arrows** step a week,
**C** compare, **U** uncertainty, **S** screenshot. `prefers-reduced-motion` disables auto-play and particle animation.

## What each visual shows

| visual | driven by (field in `sim_data.json`) |
|---|---|
| plant model: rice tufts, wheat stalks with ears, cotton bushes with bolls, sugarcane canes | the crop |
| plant height and width: seedling in week 1, full size by the end of the vegetative stage | `crops[c].development` (a visual curve built from the stage lengths, not measured growth) |
| golden ripening, ears and bolls appearing | `crops[c].senescence`, `development` |
| yellowing and shorter plants | accumulated `stress_0_1` (same smoothing as the greenness proxy) |
| leaning and wilting | `stress_0_1` above ~0.35 |
| scorched, paler leaves and haze | daily Tmax of the reference year plus the shift; `heat_p` |
| standing water in the paddy plots (bunds, channels fill when irrigating) | `pond_daily` (daily), `pond_mm` (weekly) |
| soil colour (pale tan when dry, dark brown when wet; blue ramp in the colour-blind palette) | `soil_moisture_0_1`, modulated per tile by its soil class |
| cracked soil | low `soil_moisture_0_1` |
| terrain relief, terraced plots for paddy | `districts[].elevation_m` (real, see below), else `elevation_proxy` |
| rain streaks and cloud cover | the **daily** rain of the reference year x the rainfall multiplier |
| sky, sun position, light colour | `tmax` of the day, rain, and the time-of-day control (illustrative) |
| blue sprinkler particles (upland crops) | `irrigation_mm` of the week |
| three strips with low / model / high vigour (uncertainty view) | `yield.lo`, `yield.mean`, `yield.hi` |
| water stress by growth stage panel | `stage_rel` (FAO-33 stage factors) |
| yield card | `yield.mean`, `yield.lo`, `yield.hi` |
| drought gauge | `drought_p` |
| heat gauge | `heat_p` x 7 days |
| dashed line in the greenness chart | `districts[].reference_ndvi` (labelled SYNTHETIC while the satellite dataset is) |

Everything numeric on screen comes from the file; the constants in the page are visual styling (colours, relative plant sizes,
the vertical exaggeration of the relief) and the slider ranges, which are read from `meta.grid`. Between grid points the page
interpolates bilinearly in (rainfall multiplier, temperature change). **Model state is weekly; the weather of the day is daily:**
inside a week the soil moisture, stress and yield do not change, but the date, rain streaks, sky and haze follow the day.

## What is real and what is schematic

* **Real:** the relief (elevation of a 10 km window at the district centre, open terrain tiles, shown 3x exaggerated), the daily
  weather of the reference year (IMD), and every model number.
* **Schematic:** the field layout (a 6 x 6 grid of plots with jittered lines seeded by the district id), the soil classes (8 x 8
  tiles, procedural), the plants (enlarged and generic), the sky, clouds and the time-of-day light, the depth of the paddy water
  (not to scale), and the greenness proxy.
* **Not modelled, so not drawn:** pests, lodging, flowering sterility from heat, weeds, real soil maps. The leaves of a heatwave
  week are drawn scorched, but the yield model treats temperature separately and the page says so.

## Data contract (`sim_data.json`, schema version 2, validated with `simulator/sim_schema.json`)

```text
{
  "meta": { schema_version: 2, generated_at, git_commit, weeks: 16, attribution[],
            is_synthetic_inputs: { weather, crop_labels, satellite_dataset, satellite_used_by_served_models },
            model_versions: { yield, drought, heatwave_early_warning, water_balance, irrigation },
            risk_levels, grid: { rain_mult[7], temp_delta[6], strategies[3] }, definitions{}, disclaimer },
  "crops": { "<crop>": { season, season_start "MM-DD", note, stage_days[4], ponded, pond_max_mm, percolation_mm_day,
                         initial_pond_mm, development[16], senescence[16] } },
  "districts": [ { id, name, state, cluster, cluster_label, lat, lon, area_km2,
                   elevation_proxy[32][32], elevation_m[32][32]?, mean_elevation_m?, relief_m?, elevation_source,
                   tiles[8][8], crops[], reference_year{crop},
                   reference_daily{crop: {rain_mm[112], tmax_c[112], tmin_c[112]}},
                   reference_ndvi{crop: null | {weekly[16], is_synthetic}},
                   budget_mm{crop}, taw_mm{crop}, early_warning{season: risk_forecast result} } ],
  "scenarios": { "<district>|<crop>|<rain_mult>|<temp_delta>|<strategy>": {
        "yield": { mean, lo, hi },            // t/ha, 80% interval
        "water_used_mm": x,                   // net irrigation over the season
        "stage_rel": [4 numbers],             // FAO-33 stage factors, 1 = no loss
        "pond_daily": [112 numbers]?,         // paddy only: standing water per day (mm)
        "weekly": [ { week, rain_mm, tmax_c, soil_moisture_0_1, ndvi, etc_mm, eta_mm,
                      irrigation_mm, excess_mm, stress_0_1, pond_mm, seepage_mm, drought_p, heat_p } x 16 ] } }
}
```

`strategy` is `rainfed`, `baseline` (50 mm every 10 days) or `optimized` (the Phase 6 weekly LP). Keys use Python `:g`
number formatting (`0.4`, `1`, `-2`). Floats are rounded to 3 decimals; the file is ~11 MB (limit 15 MB).

### Where each number comes from

* **yield**: `predict_yield` (Phase 4) at *normal rainfall* and the scenario temperature, multiplied by the water-stress ratio of the daily
  water balance relative to the farmer baseline (docs/DECISIONS.md D-020).
* **soil moisture, ETc/ETa, irrigation, stress, excess, stage factors**: the daily FAO-56 water balance (Phase 6) scored per strategy; the
  optimized irrigation is the weekly LP.
* **pond and seepage**: the ponded-field extension of the water balance for paddy rice (D-029): standing water behind 75 mm bunds, seepage
  of 2 mm/day while water stands, 25 mm at the start. The values are approximate and unverified.
* **`drought_p`**: the Phase 5 drought model re-evaluated on scenario rainfall (SPI distributions fixed on the reference years).
  Temperature does not change it (SPI drought is defined by rainfall).
* **`heat_p`**: the share of the week's days that qualify as IMD heatwave days under the temperature shift. It is a **deterministic
  indicator, not a model probability**: the served heatwave probability is climatology (D-019) and cannot react to a scenario.
* **`ndvi`**: a **visual proxy** from the crop-coefficient curve and accumulated water stress. It is *not* satellite NDVI.
* **`reference_ndvi`**: the satellite NDVI of the reference year (the 16-day composite that covers each week's last day). It carries
  `is_synthetic`; while the satellite dataset is synthetic the page draws it as a dashed line labelled SYNTHETIC and never lets it drive the
  plants. When the dataset is real (after Earth Engine is authenticated and `make etl` is re-run) the flag becomes false, the legend says
  "observed" and a choice "Observed NDVI of the reference year" appears. Observed NDVI does not react to the sliders.
* **`elevation_m`**: elevations (m) on a 32 x 32 lattice over a 10 km window at the district centre (row 0 = north), from open terrain tiles
  (SRTM and other public data via AWS Terrain Tiles; attribution in `meta.attribution` and in the footer).
* **`reference_daily`**: observed daily rain, Tmax and Tmin of the reference year for the 16-week window, before the scenario changes.
* **`early_warning`**: `risk_forecast` (Phase 5) on observed weather of the reference year; it does not change with the sliders.

## Known limits

* A rainfed crop can collapse to near-zero yield in the model: the FAO-33 multiplicative stage response is harsh when ETa/ETc is
  low, and the parameters are approximate (config/crop_params.yaml).
* Against the plan's generous 600 mm baseline the optimized schedule saves water and keeps slightly less yield; the page reports what
  the data say rather than the story in the plan's demo script. With ponded rice the figures changed from the first version
  (D-029); `reports/prescriptive_results.md` has the current ones.
* The paddy model leaves out land preparation (puddling) water, the nursery and bund leakage, so absolute rice water use is understated.
* Sugarcane yields are in an unclear unit (jaggery-equivalent?); compare scenarios, not absolute values.
* The vertical exaggeration (3x) makes hills look steeper than they are; the 10 km window is one place at the district centre, not the
  whole district.
* Frame rate: the 5-second measurement in `scripts/test_sim.py` runs in headless Chromium with *software* WebGL and reports the number it
  measures; it is not a measurement on integrated graphics. The page keeps per-frame work moderate (instancing, vertex-shader wind sway,
  adaptive rain particle count and plant density in automatic mode, pixel ratio capped at 2, rendering paused when the tab is hidden)
  and offers fixed High and Low detail.

## Tests

`python scripts/test_sim.py` (Playwright + Chromium; `python -m playwright install chromium` once) opens the embedded page from `file://`
and checks: no console errors; the drought gauge rises when rainfall is cut to 0.4x; compare mode shows less optimized than baseline
water; keyboard control; one plant model per crop; plants grow from seedlings; standing water only in the paddy and only when the water
balance has it; stressed plants shorter and less green; the by-stage bars show a drought loss; the date advances inside a week; the sun
is lower at 07:00; the uncertainty view draws three strips of increasing vigour; low detail draws about a third of the plants; real relief
and attribution are shown; and, on a modified copy of the data that declares the NDVI observed, the observed-greenness choice appears and
works. Screenshots go to `reports/figures/sim_*.png`; the frame rate and all measurements to `reports/sim_checks.json`.
`tests/test_simulator.py` validates the data file (schema, completeness, consistency between the weekly and daily series, the reference
weather and the scenarios), the terrain code (orientation and scale on a synthetic plane), the growth curves, the NDVI helper, the embed step
and the page's JavaScript.
