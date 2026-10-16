# 3D crop-climate simulator (Phase 7)

An **explanation tool**: a browser landscape that shows, week by week, how rainfall, temperature and an irrigation
strategy change soil moisture, crop condition and the predicted yield of a district. It visualises the outputs of the
Phase 4-6 models; it is **not a physical crop-growth simulation** and not advice for farm decisions (the same
sentence is shown in the page footer).

## Run it

```bash
python -m simulator.export_sim_data     # ~90 s: evaluates 2 772 scenarios with the Phase 4-6 code -> simulator/sim_data.json
python -m simulator.embed               # inlines the data and Three.js -> simulator/index_embedded.html
```

Then **double-click `simulator/index_embedded.html`**. It needs no server and no internet (Three.js r160.1 is inlined
from `simulator/vendor/`). `index.html` is the same page without embedded data: served over HTTP
(`python -m http.server -d simulator`) it fetches `sim_data.json`; opened from disk it shows a drop zone for the JSON file.
`index.html?district=<id>&crop=<crop>` preselects a district and crop (used by the Streamlit page).

Controls: district and crop selectors, rainfall multiplier (0.4-1.6), temperature change (-2 to +4 C), irrigation strategy
(rainfed / farmer baseline / optimized), week scrubber with play/pause and speed, compare mode (baseline vs optimized,
cameras synchronised), screenshot, colour-blind-safe soil palette. Keys: **Space** play, **arrows** step a week,
**C** compare, **S** screenshot. `prefers-reduced-motion` disables auto-play and particle animation.

## What each visual shows

| visual | driven by (field in `sim_data.json`) |
|---|---|
| rain streaks and cloud cover | `rain_mm` of the week |
| soil colour (tan to blue) | `soil_moisture_0_1`, modulated per tile by its soil class |
| cracked soil | low `soil_moisture_0_1` |
| flood plane | recent `rain_mm` on a saturated soil |
| crop height and greenness | `ndvi` (proxy) and the stress state |
| wilting and yellowing | `stress_0_1` above ~0.35, strong above 0.6 |
| blue sprinkler particles | `irrigation_mm` of the week |
| sky and sun tint | `tmax_c`, plus `heat_p` |
| yield card | `yield.mean`, `yield.lo`, `yield.hi` |
| drought gauge | `drought_p` |
| heat gauge | `heat_p` x 7 days |

Everything numeric on screen comes from the file; the only constants in the page are visual styling (colours, relative
plant sizes) and the ranges of the sliders, which are read from `meta.grid`. Between grid points the page interpolates
bilinearly in (rainfall multiplier, temperature change).

## Data contract (`sim_data.json`, validated with `simulator/sim_schema.json`)

```text
{
  "meta": { schema_version, generated_at, git_commit, weeks: 16,
            is_synthetic_inputs: { weather, crop_labels, satellite_dataset, satellite_used_by_served_models },
            model_versions: { yield, drought, heatwave_early_warning, water_balance, irrigation },
            risk_levels, grid: { rain_mult[7], temp_delta[6], strategies[3] }, definitions{}, disclaimer },
  "crops": { "<crop>": { season, note, stage_days[4] } },
  "districts": [ { id, name, state, cluster, cluster_label, lat, lon, area_km2,
                   elevation_proxy[32][32], tiles[8][8], crops[], reference_year{crop}, budget_mm{crop},
                   taw_mm{crop}, early_warning{season: risk_forecast result} } ],
  "scenarios": { "<district>|<crop>|<rain_mult>|<temp_delta>|<strategy>": {
        "yield": { mean, lo, hi },            // t/ha, 80% interval
        "water_used_mm": x,                   // net irrigation over the season
        "weekly": [ { week, rain_mm, tmax_c, soil_moisture_0_1, ndvi, etc_mm, eta_mm,
                      irrigation_mm, excess_mm, stress_0_1, drought_p, heat_p } x 16 ] } }
}
```

`strategy` is `rainfed`, `baseline` (50 mm every 10 days) or `optimized` (the Phase 6 weekly LP). Keys use Python `:g`
number formatting (`0.4`, `1`, `-2`). Floats are rounded to 3 decimals; the file is ~9 MB (limit 15 MB).

### Where each number comes from

* **yield**: `predict_yield` (Phase 4) at *normal rainfall* and the scenario temperature, multiplied by the water-stress ratio of the daily
  water balance relative to the farmer baseline (docs/DECISIONS.md D-020).
* **soil moisture, ETc/ETa, irrigation, stress, excess**: the daily FAO-56 water balance (Phase 6) scored per strategy; the optimized
  irrigation is the weekly LP.
* **`drought_p`**: the Phase 5 drought model re-evaluated on scenario rainfall (SPI distributions fixed on the reference years).
  Temperature does not change it (SPI drought is defined by rainfall).
* **`heat_p`**: the share of the week's days that qualify as IMD heatwave days under the temperature shift. It is a **deterministic
  indicator, not a model probability**: the served heatwave probability is climatology (D-019) and cannot react to a scenario.
* **`ndvi`**: a **visual proxy** from the crop-coefficient curve and accumulated water stress. It is *not* satellite NDVI (the
  satellite dataset is synthetic until Earth Engine is authenticated and the served models do not use it).
* **`elevation_proxy`, `tiles`**: procedural, seeded by the district id; **not** real relief or soil maps.
* **`early_warning`**: `risk_forecast` (Phase 5) on observed weather of the reference year; it does not change with the sliders.

## Known limits

* A rainfed crop can collapse to near-zero yield in the model: the FAO-33 multiplicative stage response is harsh when ETa/ETc is
  low, and the parameters are approximate (config/crop_params.yaml).
* Against the plan's generous 600 mm baseline the optimized schedule saves water but usually yields a little less in dry scenarios;
  the page reports what the data say rather than the story in the plan's demo script.
* Sugarcane yields are in an unclear unit (jaggery-equivalent?); compare scenarios, not absolute values.
* Frame rate: the 5-second measurement in `scripts/test_sim.py` runs in headless Chromium with *software* WebGL and reports the
  number it measures; it is not a measurement on integrated graphics. The page keeps per-frame work small (instancing, reused
  buffers, adaptive rain particle count, pixel ratio capped at 2, rendering paused when the tab is hidden).

## Tests

`python scripts/test_sim.py` (Playwright + Chromium; `python -m playwright install chromium` once) opens the embedded page from `file://`
and checks: no console errors, the drought gauge rises when rainfall is cut to 0.4x, compare mode shows less optimized than baseline
water, keyboard control works, screenshots at weeks 3, 9, 14 (`reports/figures/sim_week*.png`), and prints the frame rate.
`tests/test_simulator.py` validates the data file, the embed step and the page's JavaScript syntax.
