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

## 2026-10-16 - Phase 1 (ETL)

### D-006: Study window, splits and district units follow what real data can support
- **ICRISAT's district database ends at crop year 2017** (API `years` = 1966..2017, checked 2026-10-16),
  so real yields cannot cover the plan's 2000-2024. Rather than fabricate 2018-2024 yields, the yield
  modelling window is 2000-2017 and the yield time split is shifted to **train <= 2011, validation
  2012-2014, test 2015-2017** (`config/settings.yaml: splits.yield`). The plan's 2022-2024 test years
  are not possible with real labels. Users can extend later by placing a data.gov.in APY CSV (to ~2020)
  in `data/raw/crops/`.
- Weather is kept for **1991-2024** because Phase 5 needs a 1991-2020 climatology and 2020-2024 test years.
  Yields are kept from 1991 (`years.history_start`) so lag features exist for 2000.
- **Crop year Y** = the season starting in calendar year Y (rabi runs Nov Y - Mar Y+1), ICRISAT's convention.
- **Districts are yield-reporting units, not raw 2021 ADM2 polygons.** ICRISAT's apportioned series is stated
  on 1966 parent districts (51 units in our 4 states: TN 12, AP 11, Telangana 9, Karnataka 19), while geoBoundaries
  has 114 modern polygons. Each unit = the modern polygons whose names match the ICRISAT district (alias map),
  plus the remaining polygons attached to the nearest anchor, with 11 hand-reviewed overrides for known splits
  (`config/unit_overrides.csv`). This is a *reconstruction* of the old boundaries, so unit polygons are approximate.
  ICRISAT's own coordinates were not used for anchoring because two are wrong (Chengalpattu at 24.6N, South Arcot at 9.3N).
- Yanam (a Puducherry enclave inside Andhra Pradesh) is excluded; geoBoundaries 2021 predates AP's 2022 split
  into 26 districts, which suits the 1966-parent units anyway.

### D-007: Season and unit assumptions in ICRISAT yields
- ICRISAT gives annual totals per crop, not kharif/rabi splits. Each crop is attributed to one season
  (rice kharif, wheat rabi, cotton kharif, sugarcane kharif; `settings.yaml: crop_season`). Rice is also
  grown in rabi/summer in AP, TS and TN, so "kharif rice" is really an annual rice average.
- ICRISAT sugarcane "production" gives ~8 t/ha against usual cane yields of ~70-100 t/ha, so it is probably
  jaggery (gur) equivalent. We have not verified this against ICRISAT documentation; treat sugarcane as a
  relative index and never compare it with cane tonnage from other sources.
- Cotton yield (~0.35 t/ha) is lint. Missing crop = area 0 (ICRISAT does not use NaN); those rows are dropped.

### D-008: Soil moisture from ERA5-Land instead of SMAP; GEE not yet authenticated
SMAP L4/HSL soil moisture starts in April 2015, so it cannot cover the 2000-2014 training years. The GEE module
uses `ECMWF/ERA5_LAND/DAILY_AGGR` layer 1 (m3/m3) averaged over each MODIS 16-day window. Earth Engine is **not
authenticated on this machine**, so `satellite_16d` is currently **synthetic** (derived from real rainfall by
construction, `is_synthetic=True`). Everything that uses NDVI/soil moisture must say so until
`python -m etl.gee_auth_check` passes and `python -m etl.run --source gee --force` is run.

### D-009: NASA POWER as a real fallback before synthetic
POWER daily point data (free, no key) supplies humidity, wind and solar radiation, and also temperature and
precipitation. If an IMD year cannot be downloaded (imdpune.gov.in times out regularly), POWER fills it with
real (model-derived, ~0.5 degree) data and the share is reported in `docs/DATA_QUALITY.md`. Synthetic generators
are the last resort only.

### D-010: ICRISAT access
The DLD web portal is an interactive app, but it is a thin client over `http://data.icrisat.org/dldAPI/`
(JSON). We call `apportioned/area-production-yield` once and cache the response in `data/raw/crops`. No scraping
of rendered pages. Cite "ICRISAT District Level Database" and check its terms of use before redistributing data.

## 2026-10-16 - Phase 2 (warehouse)

### D-011: No partitioning; lead-specific columns live in the view
- `weather_daily` has ~0.63 M rows (51 units x 34 years), so the plan's "partition by year if over 5M rows"
  does not apply. The primary key (district_id, date) is the btree index the plan asks for.
- The plan's `v_training_table` description only mentions full-season aggregates, but Phase 4 forecasts at 30/60/90-day
  leads and full-season weather would leak the future. `v_season_weather` therefore also emits `*_d30/_d60/_d90`
  columns (season days 0..L-1) and satellite values from composites that had finished by the forecast date
  (MODIS composite `d` covers `d..d+15`). Leakage-safety of every column is documented in `database/README.md` and
  asserted in `tests/test_database.py`.
- `crops_annual` key is (district_id, year, season, crop): the plan says "PK over all five" but only four key columns exist.

## 2026-10-16 - Phase 3 (analytics)

### D-012: Reliability filter for ICRISAT crop rows
Trend plots exposed source errors: Tamil Nadu cotton 2017 reports 10-30 t of production on 19 000-60 000 ha (yield
0.0005 t/ha; ICRISAT's own reported yield is equally absurd, so the reported-vs-recomputed `yield_flag` cannot catch it),
Tumkur cotton 2016 gives 2.3 t lint/ha and Adilabad wheat 2015 gives 4.9 t/ha. `etl/crop_yield.py: filter_unreliable`
now drops rows where (1) reported and recomputed yield disagree by more than 50 percent or (2) the yield is outside a
plausible per-crop range (`settings.yaml: etl.min/max_plausible_yield_t_ha`; sugarcane has no bound because its unit is
unclear, D-007). 7 of 4 293 rows (0.16 percent) were dropped; they are listed in `docs/DATA_QUALITY.md`. The bounds
are physical/agronomic limits chosen before looking at model results, not tuned on the test years.

## 2026-10-16 - Phase 4 (yield models)

### D-013: Serving model uses weather features only
NDVI / soil moisture cannot be recomputed when a what-if changes rainfall or temperature, and they are synthetic
(derived from rainfall) until Earth Engine is authenticated. `predict_yield` therefore serves an XGBoost trained on the
`weather` feature set; `weather+satellite` models are trained and reported for comparison only. Overrides are applied
to the *daily* weather of a reference year and all season features are recomputed with the same code as the SQL
view (`analytics/predictive/features.py`, checked against the view in tests), so scenarios stay internally consistent.

### D-014: Temperature features are monotone-constrained in the shipped XGBoost
The first full run gave a what-if response where +4 C *raised* forecast yield by ~9 percent on average (96 percent of
district-years), because trees cannot extrapolate beyond the historical temperature range. Phase 3's fixed-effects
regressions show heat lowering rice and cotton yield, so XGBoost (point and quantile models) now carries monotone
non-increasing constraints on mean Tmax, heat-day and degree-day features. Cost: validation relative RMSE 0.342 vs
0.339 unconstrained, test 0.410 vs 0.404 (reported in `reports/yield_model_comparison.md`). The constraint is a domain
assumption (it would be wrong for cool highland districts at some temperatures), not a learned fact.

### D-015: Baselines and what the models achieve
Besides the plan's naive last-year baseline, two weather-free baselines were added: a per-district linear trend (planned
before the first run) and the previous-3-year mean (added after the first evaluation round, when the first test numbers
had been seen; it can only make claims more conservative). Result on the test years 2015-2017: the models beat naive last
year clearly (about 17 percent lower relative RMSE) but are **statistically indistinguishable from the trend and 3-year-mean
baselines**, consistent with Phase 3 (weather adds very little beyond district and year effects). Validation-year skill
(relative RMSE ~0.34) did not carry over to the test years (~0.41), so the numbers should not be quoted without the baselines.
The plan's hypothesis targets (RMSE 2.9, R2 0.82, F1 0.75) are not reproduced and are not claimed.

### D-016: Serving models are fitted on train years only
The shipped models are the ones that were evaluated (fitted on crop years <= 2011, tuned on 2012-2014), not refit on
validation or test years, so reported metrics describe exactly what `predict_yield` runs. Conformal intervals are
calibrated on 2012-2014; test coverage is slightly below the 80 percent target (see the report).

### D-017: IMD is the primary weather source; POWER differs mainly in extremes
After the IMD download completed (all 34 years of rain, Tmax and Tmin), rain, Tmax and Tmin are 100 percent IMD and only
humidity, wind and solar come from NASA POWER. Comparing the two on the same districts: daily rain correlation 0.61,
annual district rain correlation 0.81, daily Tmax correlation 0.85 (POWER 0.19 C cooler on average), but **heat days (Tmax >= 40 C)
average 10.6 per district-year in IMD versus 21.9 in POWER**. Results that were first produced from the POWER stand-in
(Phase 3 findings, Phase 4 first runs) were regenerated on IMD before the phases were tagged.

## 2026-10-16 - Phase 5 (extreme-weather early warning)

### D-018: Reference period and evaluation design for rare events
- **Reference period 1991-2014 (training years), not the plan's 1991-2020**, for the heatwave daily normals and the SPI gamma fits,
  so that no fitted quantity (and hence no feature or label threshold) depends on validation or test years. Cost: a 24-year
  rather than 30-year normal, and a reference cooler/drier than 2020-2024, which shifts base rates between splits.
- **Heatwave onsets are extremely rare** with the IMD criterion on 1-degree district-mean data: 108 onsets in 34 years x 51 districts
  (~0.06 percent of days are heatwave days); the plan's test window 2020-2024 has only 3 onsets. Besides the plan's fixed split
  (final fold), skill is therefore judged on pooled **rolling-origin folds** (always train on earlier years, validate on the 3-5 years
  before the test block; test blocks 2005-09, 2010-14, 2015-19, 2020-24 give 37 pooled onsets). Block bootstrap (district-years)
  supplies the confidence intervals.
- Drought uses the plan split only (its features use the reference-period SPI, so rolling folds would leak). Test years 2020-2024
  are much wetter than the reference: next-month drought base rate 15.8 percent (train), 20.9 percent (validation), 7.5 percent (test).

### D-019: The heatwave LSTM does not beat climatology, so climatology is served
Pooled out-of-sample, the LSTM is *worse* than a district-by-calendar-day climatological frequency at all three horizons (PR-AUC,
Brier skill); the 14-day lead is therefore not supported. `risk_forecast` returns the climatological probability for heatwave
(flagged `heatwave_source="climatology"`) and switches to the LSTM automatically only if its 14-day PR-AUC advantage has a
95 percent bootstrap interval above zero when retrained. The drought model does beat climatology and persistence (PR-AUC 0.38 vs
0.09 / 0.16 on the test years). Per-hazard risk thresholds replace the single shared set because base rates differ by ~20x.

## 2026-10-16 - Phase 6 (prescriptive optimization)

### D-020: How the ML yield and the water balance are combined (no double counting)
The ML yield model already responds to rainfall, so multiplying its output by a water-stress factor would count drought twice.
`scenario.strategy_yields` therefore uses the ML prediction at **normal rainfall** (scenario temperature only) and scales it by
`RYF(strategy, scenario) / RYF(farmer baseline, normal rainfall)`, so the farmer baseline at normal rainfall reproduces the ML
yield, droughts lower it, better irrigation raises it, and the gain is capped at +50 percent. This is a modelling convention,
not an estimate validated against irrigated-yield data; the plan's "ML potential x relative yield factor" is followed in spirit.

### D-021: Weekly LP linearisation and its limits
The LP uses a piecewise-linear stress constraint. The first version (start-of-week storage) ignored depletion *inside* a
week and under-irrigated: scored by the daily simulator it lost to an even split. Judged on 42 simulated scenarios (not real
labels) the **end-of-week storage basis** was best (optimized relative yield 0.860 vs even split 0.812 with less water; optimized >=
even split in 76 percent of scenarios vs 57 percent for the start basis), so it is the default. The guarantee "LP never worse than the
even split" holds under the LP's own model (tested via its penalised optimum), **not** under the daily simulator: in the 20-case
report the optimized schedule is worse than the even split in 8 (normal rain) and 10 (dry) cases while using ~40 percent less water.
Parameters in `config/crop_params.yaml` are approximate and flagged as such; rice is not modelled as a ponded system.

### D-022: Economics inputs are illustrative; sugarcane excluded
MSP/FRP-order prices and costs in `config/economics.yaml` were entered from memory of public notices and are unverified. Source
data units need conversion (ICRISAT rice is milled rice, MSP is for paddy: /0.67; cotton is lint, MSP is for seed cotton: /0.34).
Sugarcane is excluded from the comparison because its yield unit looks like jaggery equivalent (D-007). Outputs carry a
"decision-support estimate, not financial advice" disclaimer.

## 2026-10-16 - Phase 7 (3D simulator)

### D-023: What the simulator shows, and where fields that no model produces come from
The simulator needs weekly values *for what-if scenarios*; several of them do not exist as model outputs, so each is defined explicitly
in `meta.definitions` and the README:
- `ndvi` is a **visual proxy** (crop-coefficient canopy x accumulated water stress), not satellite NDVI: the satellite dataset is synthetic
  and the served models do not use it. The HUD labels it "greenness proxy (not satellite NDVI)".
- `drought_p` is the Phase 5 drought model re-evaluated on scenario rainfall with the SPI reference distributions held fixed (new
  `analytics/predictive/risk_scenarios.py`; SPI code refactored into fit/apply without changing outputs). Temperature does not
  affect it: an earlier test showed +3 C *lowering* drought risk, a tree-extrapolation artefact of the Tmax-anomaly input.
- `heat_p` is the share of days qualifying as IMD heatwave days under the temperature shift: a deterministic indicator, because the
  served heatwave probability is climatology (D-019) and cannot react to a slider. The gauge says "heatwave days", not "probability".
- Elevation and soil tiles are procedural (seeded by the district id), not SRTM or soil maps.
- Three.js comes from jsDelivr (cdnjs does not host the r160 module build), is vendored in `simulator/vendor/` (MIT, SHA recorded) and
  inlined by `python -m simulator.embed` so the page works offline from `file://`.
- The plan's demo narrative ("optimized ... yield recovers") is **not** what the models show against the 600 mm baseline: the optimized
  schedule saves water but usually yields slightly less. The simulator displays the model output; the Phase 10 demo script must follow
  measured results.
- Frame rate is measured in headless Chromium with software WebGL (about 15 fps at 1500x900); this is not a measurement on integrated
  graphics and the 30+ fps acceptance criterion is **not verified here**. The code limits per-frame work (instancing, reused buffers,
  adaptive rain count), but only a run on real hardware can confirm the criterion.

## 2026-10-16 - Phase 8 (dashboards) and advisory groundwork

### D-024: Dry-season drought signals do not set the risk level or the advisory
A test advisory for Krishna on 2005-01-10 raised a "severe drought" alert because the SPI of a month that normally gets almost no
rain is dominated by tiny absolute deficits. `risk_forecast` now returns `drought_relevant` (False when the target month's
normal rain is below `early_warning.drought_min_month_rain_mm`, 40 mm) and `drought_normal_month_rain_mm`; the probability is still
reported, but it is excluded from `level`, and `advisory.generate` suppresses the drought watch with an explanatory note. Added keys
only: the earlier contract (`heatwave_p`, `drought_p`, `level`, `lead_days`) is unchanged. The 40 mm threshold is a judgement call,
not a fitted value.

### D-025: Dashboard design choices
- **Router.** `dashboards/app.py` is a thin `st.navigation` router (so the sidebar says "Home" rather than "app"); every page is a
  stand-alone script that bootstraps `sys.path`, calls `layout.start` (page config, data guards, synthetic banner, shared
  district/crop sidebar) and wraps its body in `guards.run_guarded`, which turns missing files and `ValueError`s into messages.
- **Altair, not Plotly/Folium, for charts and maps.** Streamlit ships Altair, so the dashboards need no extra dependency and the
  district choropleths (`mark_geoshape`) work offline. Folium/Plotly stay in the analytics reports only.
- **"Weather forecast" is a replay.** There is no live forecast feed. The page runs the Phase 5 models on observed weather up to a
  chosen date within the record (1991-2024) and says so on the page.
- **Vulnerability index** (Risk map, second tab) is defined here, not in an earlier phase: the mean percentile rank of kharif drought
  frequency (season rain < 75 percent of the district mean), rice-yield coefficient of variation (2000-2017, at least 10 years) and
  mean heat days (Tmax >= 40 C) per kharif season. It describes history and is not a forecast.
- **Headline accuracy tile compares with the best simple baseline** on relative RMSE (error as a share of each district's prior mean
  yield, the quantity the models optimise and the bootstrap uses), not with the weakest baseline. In absolute t/ha the pooled RMSE is
  dominated by sugarcane and the prior-three-year mean is actually lower than the model's, which the Methodology table shows.
- **What-if sensitivity is weak and the page says so.** The served model moves the forecast by about -2 percent for 40 percent less
  rain and -3.5 percent for +4 C (validation years); it learned yield mostly from each district's history.
- **Simulator embedding.** `simulator/index_embedded.html` is copied on demand to `dashboards/static/simulator.html` (git-ignored)
  and shown with `st.iframe("/app/static/simulator.html?district=..&crop=..")`. The simulator holds only six representative districts
  (one per climate cluster, none for cluster 1), so other selections are mapped to the same-cluster representative or the nearest
  district, with a visible note.
- **Browser checks.** `scripts/screenshot_dashboards.py` (Playwright) reloads every page, requires no exception boxes or failed
  requests, takes the median of three warm visits (budget 3 s; measured 1.3-2.4 s in software-rendered headless Chromium, server-side
  script time 0.3-0.9 s) and writes `reports/dashboard_checks.json` and `reports/figures/dash_*.png`. Streamlit's own deep-link probes
  of `/<page>/_stcore/health` and `/host-config` return harmless 404s and are ignored.

### D-026: Translations are unreviewed drafts
Hindi, Tamil and Telugu templates in `advisory/templates.yaml` were written without a native-speaker or agronomist review. The
dashboard warns on every non-English advisory and the sender defaults to dry run (Phase 9). Review is a manual step before any real send.

## 2026-10-16 - Phase 9 (advisory delivery)

### D-027: Delivery and bot are safe by default; the bot answers about the last observed day
- **Dry run unless explicitly disabled.** Only `DRY_RUN` set to false/0/no/off leaves dry-run mode (any other value, including a typo,
  stays dry). Real sending additionally needs the account SID, auth token and sender number, otherwise `ConfigError` is raised
  before a client exists. The Twilio library is only imported on the real-send path.
- **The webhook is closed by default.** Every `/whatsapp` call needs a valid `X-Twilio-Signature`; with no auth token configured
  the endpoint answers 403 unless `ALLOW_UNSIGNED_WEBHOOK=true` (local testing). `PUBLIC_BASE_URL` is used to rebuild the URL Twilio
  signed when the bot sits behind a tunnel or proxy. The reply is TwiML in the webhook response, so `DRY_RUN` (which governs
  outbound API sends) does not apply: the bot only answers people who wrote to it.
- **Uncertain districts are never used silently.** A fuzzy match is accepted only at a score of at least 92 (rapidfuzz, whole-word
  containment counts as 95) and at least 4 points ahead of a different district; between 70 and 92, or when two districts are
  close ("godavari"), the bot asks. Thresholds are in `settings.yaml` (`advisory.bot`) and are judgement calls checked on a few
  dozen spellings, not tuned on data.
- **State is minimal.** Conversation memory (district, crop, language) is in process memory for 30 minutes, keyed by a hash of the
  number. The only persisted personal data is the opt-out list (salted SHA-256 hashes) and an audit log of masked numbers and
  message lengths, never message bodies or tokens. STOP is honoured by the bot, the sender and the bulk tool.
- **No live weather feed exists in the project**, so without a date in the message the bot advises for the last day of the observed
  record (`ADVISORY_AS_OF` overrides). This is stated in the README; a real deployment needs a forecast source.
- Control replies (help, confirmation questions) are English only; district names must be typed in English letters. Both are
  limitations of this version.
- Bulk composition never sends; it writes a reviewable CSV with masked numbers.

## 2026-10-16 - Phase 10 (hardening, packaging, report)

### D-028: Packaging, demo bundle and generated documents
- **Demo bundle in git.** `data/sample/` (git-tracked; the rest of `data/` and `models/` is ignored) holds the derived district tables,
  the three serving models and the simulator page with a SHA-256 manifest (`python -m config.bundle make|restore|verify`). Dashboards
  call `restore_if_missing()` at start-up, so a fresh clone, a container or a hosted Space runs without the multi-hour IMD download.
  It contains IMD-derived series and ODbL-derived polygons whose redistribution terms are **unconfirmed**: `scripts/check_licenses.md`
  lists what to verify, and the bundle should be dropped from a public push if permission is not obtained. The bundle's satellite table
  stays flagged synthetic.
- **Two dependency files.** `requirements.lock` is the exact development environment (the models are pickles and need the same
  scikit-learn/XGBoost versions); `requirements.txt` is the exact serving subset with the CPU PyTorch index, used by the Docker image
  and by Streamlit Cloud. `tests/test_packaging.py` keeps them consistent with each other and with `pyproject.toml`.
- **Image.** `python:3.11-slim`, non-root user, health check, CPU-only PyTorch (default Linux wheels pull CUDA libraries that a CPU
  service never uses), dependencies in a separate layer.
- **Generated documents cannot drift from the results.** `scripts/make_report.py` writes the final report, the README results block,
  the demo script and the CV bullets from `reports/*.json`; unknown values print as `MISSING (...)`, and `verify_claims` fails the run
  when a sentence of the fixed prose (heatwave LSTM has no skill, optimizer keeps less yield, the yield model is not clearly better
  than baselines, the satellite table is synthetic) is contradicted by the numbers, so the prose must be rewritten rather than
  silently left wrong. The plan's targets are shown as hypotheses with measured values beside them, never as achievements.
- **`make pipeline`** chains db-up, etl, db-all, analytics-eda, train-yield, train-extreme, prescriptive-report, export-sim, embed-sim
  and stops at the first failure; every step is idempotent.

## 2026-10-16 - Simulator realism upgrade (v1.1)

### D-029: Paddy rice is a ponded field; results of the irrigation phase changed
The earlier water balance treated rice like any root-zone crop (D-021 noted that seepage and ponding were missing). The model now has an
optional ponded-field extension, enabled for rice in `config/crop_params.yaml` (`paddy:`): depletion may go negative down to
`-pond_max_mm` (standing water behind the bunds, 75 mm), `min(percolation_mm_day, standing water)` (2 mm/day) is lost to seepage each day
while water stands, water above the bunds spills, and the field starts with 25 mm of water. Without the block, or for any other crop, the
balance is bit-for-bit the earlier one (tested). The weekly LP gets the matching storage bound (`TAW + pond`) and a weekly seepage loss that
cannot exceed the water available (a penalised slack keeps it feasible when water is short). The three values are **approximate and
unverified**, like the other agronomic parameters. Land preparation (puddling) water, the nursery and bund leakage are still not modelled.
- The rice seasonal budget was raised from 350 to 500 mm (a rounded allowance for seepage); other crops are unchanged.
- **Consequences, all regenerated from the new model:** `reports/prescriptive_results.*`, the simulator data, the irrigation amounts in
  rice advisories, and the final report. Optimized irrigation now saves 61 percent of the 600 mm baseline water in normal rainfall and keeps
  98 percent of the baseline yield (was 70 and 93 percent); against an even split it is a small but clear gain in normal rainfall (interval
  +0.0005 to +0.027) and not clear in the dry scenario. The report generator now computes that verdict from the interval instead of fixed
  prose (`analytics/prescriptive/wording.py`), and its claim check had caught the old sentence ("not distinguishable") becoming wrong.
- A ponded field buffers water that a root-zone model would discard above field capacity, so with the same schedule the ponded rice can
  yield *more* than the earlier model while losing seepage: ponding is not simply "more demanding".

### D-030: What the simulator draws from data and what it invents (schema version 2)
The scene was made more realistic without making it look more certain than it is. New data fields: `pond_daily`, `pond_mm`, `seepage_mm`,
`stage_rel` (FAO-33 stage factors), `reference_daily` (observed daily weather of the reference year, unscaled; the page applies the sliders),
`reference_ndvi` (flagged synthetic), `elevation_m` and `relief_m` (real relief of a 10 km window from open terrain tiles, with attribution),
and per-crop `development` / `senescence` (a visual growth curve built from the stage lengths).
- **Real relief, schematic everything else.** Elevation comes from AWS Terrain Tiles (SRTM and other public data; cached under
  `data/raw/elevation`; falls back to the procedural proxy, labelled, when offline). Fields, soil classes and plants stay schematic and the
  page says so. Soil maps were not added (no open source was checked); the soil classes remain procedural.
- **Model state is weekly, weather is daily.** Only the standing water of paddies is exported daily (the irrigation event is visible);
  moisture, stress and yield are weekly values. The page says so.
- **Observed NDVI cannot drive the scene while the dataset is synthetic.** The dashed NDVI line is labelled SYNTHETIC; a choice "observed
  NDVI" appears only when `is_synthetic` is false (tested on a modified copy of the data). Earth Engine authentication is still a manual step.
- **No heat-sterility icon and no heat shimmer.** The yield model has no mechanism for flowering sterility, and a screen-space distortion
  was too costly for software rendering; heatwave weeks are shown as haze and scorched leaves, tied to the daily Tmax and the `heat_p` indicator.
- **Uncertainty view** illustrates the 80 percent interval with three strips of low, model and high vigour; it is not a separate model.
- The measured software-rendering frame rate fell (about 8 fps from about 12) because the scene has more geometry; the page lowers detail
  automatically below 28 fps and offers fixed High and Low. A real-GPU measurement is still missing.

### D-031: Minimal interface for the dashboards and the simulator chrome

- **One stylesheet, no framework.** `dashboards/assets/minimal.css` (injected by `dashboards/components/ui.py`) sets white surfaces, near-black
  text, one blue accent, hairline borders and no shadows. Streamlit's theme options in `.streamlit/config.toml` carry the same tokens.
- **Navigation moved to the top** (`st.navigation(position="top")`), page titles are short words, emoji icons were removed, and the toolbar,
  decoration and footer are hidden. The header is kept because it holds the navigation.
- **Notes instead of coloured boxes.** Alerts are flat with a left rule. The synthetic-data caveat is one amber-ruled line shown on every
  page while any table has `is_synthetic` rows and disappears when the share is zero. Risk levels use words (`charts.LEVEL_LABEL`), not
  coloured emoji, so they do not rely on colour alone.
- **Layout.** Reading pages use a 1040 px column; the simulator page is wide (1480 px). The simulator embeds with `?theme=light` and its own
  chrome was flattened to match; its scene and numbers are unchanged. The embedded frame is 900 px tall so the results column is not clipped.
- Nothing about the data, models or reported numbers changed. Browser checks (`make test-dashboards`) were updated for the new labels.

### D-032: "Paper and field" theme replaces the all-white look of D-031

Feedback on the first minimal version: it read as a blank white page. The structure of D-031 (top navigation, narrow column, flat notes,
no shadows beyond a hairline) stays; the colour and surfaces changed.
- **Palette in one place.** `dashboards/components/theme.py` holds the tokens; `ui.inject_css` writes them as CSS variables ahead of
  `assets/minimal.css`, and the charts and maps read the same constants. `.streamlit/config.toml` repeats the values for Streamlit's own widgets.
- **Surfaces.** Warm paper page, white cards (metrics, charts, tables, expanders, home tiles), a dark green sidebar with the product mark,
  a tinted hero on the home page, tinted notes (green for information, amber for data caveats and warnings, red for errors).
- **Type.** Headings use a serif stack (Palatino/Georgia) for an editorial feel; numbers stay in the sans face with tabular figures.
- **Chart colours were checked, not chosen by eye.** The two-series pair (rain/Tmin blue `#2a86c0`, heat/Tmax orange `#d65a2b`) passes the
  lightness, chroma, colour-vision and contrast checks of the dataviz validator. Green was rejected as a second category next to orange
  (worst protan separation 6.2, below the floor) and is used only for single series and the interface. The risk maps use one terracotta ramp.
- **Simulator chrome** takes the same tokens (paper page, white stage, green accent); the 3D scene and every number in it are unchanged.
- Nothing about data, models or reported numbers changed.

### D-033: Vercel hosts a static site; the dashboards stay on a Streamlit host

The request was to deploy the project on Vercel as "veylora". The dashboards are a Streamlit server (websocket per session, PyTorch/XGBoost/SHAP/GeoPandas,
about 45 MB of bundled data and models), which Vercel's serverless functions cannot run (no websockets, function size limit). Rewriting the dashboards
as a JavaScript front end with a separate Python API was out of proportion for a deployment request.
- **Chosen:** a static site on Vercel (`web/`, generated by `scripts/make_site.py`) with the landing page, the simulator (it runs entirely in the
  browser) and the headline results, plus the dashboards on Streamlit Community Cloud, linked with `--dashboards-url`. Until a dashboards address
  exists the page says plainly that the dashboards are not hosted there.
- **Numbers are not typed into the page.** Yield error, drought-warning F1 and the irrigation trade-off are read from `reports/*`; the comparison
  word ("better than", "about level with", "worse than") is derived from the numbers; a missing file prints `MISSING`. The synthetic-satellite note
  follows `reports/data_quality.json`.
- **Build.** `vercel.json` copies the bundled simulator page into `web/` (so the 12 MB file is not committed twice) and serves `web/` with clean
  URLs; `.vercelignore` limits the upload (checked with the `ignore` package that the Vercel CLI uses).
- **Not done here:** creating the Vercel project and the Streamlit app needs the owner's accounts. The Vercel CLI was not installed and no token was available.

### D-034: Final design is "paper and field"; the dark "almanac" alternative was dropped

Two designs were built side by side (green paper and cards on port 8501, a flat dark editorial one on 8502) and the owner chose the first. The second
design, its stylesheet, the palette switch (`CR_THEME`) and the launcher `dashboards/serve.py` were removed so only one design has to be maintained;
they remain in git history at commit `da910b4` if a dark theme is wanted later. `theme.py` now holds a single palette; the simulator's own dark
colours (used only by a browser in dark mode) were retuned from the almanac charcoal to a dark green family so the standalone page still matches the site.

