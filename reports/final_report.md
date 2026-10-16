# ClimateResilience-India: final report

*Generated 2026-10-16 by `scripts/make_report.py`. Every number below is read from `reports/*.json` and `reports/*.csv`; a number that could not be found is printed as `MISSING (...)`.*

## 1. Summary

ClimateResilience-India is a district-level decision-support system for four southern states (Andhra Pradesh, Telangana, Karnataka, Tamil Nadu). It ingests weather and crop-yield data, stores them in a PostGIS warehouse, analyses climate risk, forecasts yields, warns about heatwaves and droughts, optimizes irrigation, shows the season in a 3D simulator and delivers advice through Streamlit dashboards and a WhatsApp/SMS assistant.

**What the measurements say, plainly:**

| Area | Measured result (held-out years) | What it means |
|---|---|---|
| Data | 51 district units, 633,369 daily weather rows, 4,286 yield records; satellite table 0% synthetic | all data-quality checks passed; the served yield model uses weather features only and the served drought model uses weather features |
| Yield forecast (2015-17, 60-day lead) | relative RMSE 40.4% (pooled RMSE 1.04 t/ha, R² 0.91); best simple baseline (district trend) 40.9% | Level with the best baselines: ratio vs district trend 0.99 (95% interval 0.94 to 1.03); clearly better than last year's yield (0.81). 80% interval covers 80% of test cases |
| Drought warning (2020-24) | F1 0.50, PR-AUC 0.38 vs climatology 0.09 (base rate 7.5%) | Real skill over climatology and persistence (0.13); validation F1 was 0.70 |
| Heatwave warning (14 days) | LSTM PR-AUC 0.014 vs climatology 0.064 on 37 pooled onsets | No skill over climatology, so the **climatology** probability is served |
| Irrigation optimizer (normal rain) | saves 61% of the 600 mm baseline water; keeps 98% of baseline yield | Modelled, not field-tested. Vs an even split: 161 mm less water, yield difference +0.013 (interval +0.0005 to +0.0268), worse in 7 of 20 cases |
| Dashboards | 7 addresses, 0 failed browser checks, median warm page load 1.1-1.9 s | Headless Chromium with software rendering |
| Advisory bot | 7 of 7 smoke checks passed | Dry run by default; translations unreviewed |
| Tests | 368 automated tests | `make test`; in a clean Linux container: 327 passed, 9 skipped in 71.21s (0:01:11) |


* The yield models are **not better than simple baselines** by a margin the data can confirm. They are useful as a weather-adjusted baseline with an honest interval, not as precise predictors.
* The heatwave LSTM has **no skill over climatology**; the system serves climatology and says so. The drought model **does** beat climatology and persistence.
* The optimizer **saves a large share of the water but keeps less yield than the 600 mm baseline** in the model. Against an even split of the same kind of budget it uses less water with a clear but small yield gain. The plan's story that yield recovers while water is saved is not what the models show.
* Satellite NDVI and soil moisture are **real** (MODIS and ERA5-Land through Google Earth Engine); the served yield model uses weather features only and the served drought model uses weather features.

## 2. Data and scope

Units are the 51 yield-reporting district groups that ICRISAT publishes (1966 boundaries, reconstructed from 2021 polygons), not villages. Sources: IMD gridded rainfall and temperature, NASA POWER (humidity, wind, solar), ICRISAT district yields (2000-2017), geoBoundaries polygons. Rows: 633,369 daily weather, 4,286 yield records, 29,172 satellite rows (0% synthetic, flagged `is_synthetic`). Data-quality verdict: all data-quality checks passed (`docs/DATA_QUALITY.md`). Decisions that shape the results are logged in `docs/DECISIONS.md`.

## 3. Climate findings (descriptive and diagnostic)

1. **Rainfall variability differs sharply between districts.** The coefficient of variation of kharif rainfall averages 0.28 across 51 districts (1991-2024), against 0.71 for rabi. The most variable district is Thirunelveli (CV 0.59, mean 321 mm); the most stable is Dakshina Kannada (CV 0.14). Across districts, mean rainfall and CV have Spearman rho = -0.76 (p < 0.001). The driest regional years were 2002 (100% of districts below normal), 2003 (80% of districts below normal), 2023 (80% of districts below normal).
2. **Yield trends differ by crop and state.** Of 13 state-crop series (1991-2017, balanced district panels), 8 show a significant increase and 1 a significant decrease (Mann-Kendall, p < 0.05). Rice rises significantly in 3 of 4 states (Andhra Pradesh +0.045 t/ha/yr, +1.5%/yr, Karnataka +0.023 t/ha/yr, +0.9%/yr, Telangana +0.045 t/ha/yr, +1.6%/yr); the steepest relative gain is cotton in Telangana (+3.2% of mean per year).
3. **Kharif droughts (season rain < 75% of the district mean) hit about one season in 6.** Mean frequency is 18.1% of 34 seasons; Thirunelveli is most exposed (11 events, 32%). The worst regional years were 2002 (73% of districts), 2016 (53% of districts), 2003 (49% of districts). Under the strict flood-risk definition (> 125% of mean and >= 3 days > 100 mm of district-mean rain) there were only 11 events in 9 districts, because district-area-mean rainfall rarely exceeds 100 mm/day.
4. **Pooled and within-district views of weather and yield can disagree.** Pooled across districts, rice yield and kharif rainfall have Pearson r = -0.17 (95% cluster-bootstrap CI -0.33 to -0.02); within districts, after removing each district's level and trend, r = 0.19 (0.11 to 0.28). The sign flips: wetter districts yield less rice, but a wetter year helps a given district (spatial confounding, so pooled correlations are misleading). In the two-way fixed-effects regression (district and year effects, district-clustered SEs, n = 918), +100 mm of rainfall is associated with higher yield (+0.024 t/ha, p < 0.001) and +1 C of mean Tmax has no statistically detectable association with yield (-0.107 t/ha, p = 0.070); in the district-FE-plus-trend model Tmax is -0.147 t/ha (p < 0.001) and rain +0.022 t/ha (p = 0.003). Rain and Tmax together add 0.6% of R-squared beyond district and year effects, which sets a low ceiling for any weather-based yield forecast.
5. **Districts fall into 7 climate groups** (KMeans, silhouette 0.37, features from years <= 2011 only): dry-hot-variable-rain-long-dry-spells (3 districts, mean kharif rain 431 mm, e.g. Madurai, Thirunelveli); dry-hot-variable-rain-long-dry-spells-volatile-yield (2 districts, mean kharif rain 456 mm, e.g. Ramananthapuram, Thanjavur); moderate-warm (15 districts, mean kharif rain 545 mm, e.g. Ananthapur, Bellary); moderate-warm-stable-rain (7 districts, mean kharif rain 746 mm, e.g. Belgaum, Bidar); moderate-hot (13 districts, mean kharif rain 834 mm, e.g. Adilabad, East Godavari); moderate-cool (9 districts, mean kharif rain 880 mm, e.g. Bangalore, Chickmagalur); wet-cool (2 districts, mean kharif rain 3155 mm, e.g. Dakshina Kannada, Uttara Kannada).

Full tables: `docs/FINDINGS.md`, `reports/analytics/`.

## 4. Yield forecasting

Target: yield relative to the district's prior mean. Time-based splits only (train to 2011, validation 2012-14, test 2015-17). Models: naive last year, district trend, prior-three-year mean, Ridge, random forest, XGBoost (Optuna, monotone temperature constraint), LSTM; 80% intervals by conformalized quantile regression.

* Test relative RMSE of the served XGBoost: 40.4%; best simple baseline (district trend): 40.9%. In absolute terms the pooled RMSE is 1.04 t/ha against 1.27 t/ha for that baseline; the best baseline by pooled RMSE is the prior mean 3y (0.97 t/ha, lower than the model's). Pooled errors are dominated by sugarcane's high yields, and R² (0.91) says little: last year's yield alone already reaches 0.88, because yields differ strongly between districts.
* Cluster-bootstrap ratios of relative RMSE (below 1 is better): vs last year 0.81; vs district trend 0.99 (0.94 to 1.03); vs prior three-year mean 0.96 (0.92 to 1.00).
* 80% interval coverage on the test years: 80%.
* What-if sensitivity of the served model (validation years): -2.2% for 40% less rain, -3.5% for +4 °C: the model learned yield mostly from each district's own history.
* Full tables and SHAP: `reports/yield_model_comparison.md`, `reports/yield_metrics.csv`, `reports/figures/yield_shap_*.png`.

## 5. Heatwave and drought early warning

* **Drought** (SPI-3 at or below -1 next month; XGBoost on rainfall-deficit features, isotonic calibration): test F1 0.50 (precision 0.41, recall 0.64), PR-AUC 0.38 against 0.09 for climatology and 0.13 for persistence; validation F1 0.70. A drought signal for a normally dry month does not set the risk level (D-024).
* **Heatwave** (onset within 14 days, March-June): pooled rolling-origin PR-AUC 0.014 for the LSTM against 0.064 for climatology over 37 onsets; the served probability is **climatology**. Onsets are rare, so conclusions are weak in both directions.
* Details: `reports/extreme_weather_report.md`.

## 6. Irrigation optimization

Daily FAO-56 water balance, FAO-33 yield response and a weekly linear programme (PuLP/CBC); the slowest LP case took 0.12 s. Paddy rice is modelled as a ponded field (standing water behind bunds and seepage while it stands; the parameters are approximate and unverified, D-029); the other crops use the plain root-zone balance.

* Normal rainfall: 61% water saved against the 600 mm baseline, yield 98% of baseline. Dry year (rain x0.7): 56% saved, yield 96% of baseline.
* Against an even split: 161 mm less water, relative-yield difference +0.013 (interval +0.0005 to +0.0268), worse in 7 of 20 cases.
* Crop parameters and prices are approximate and unverified (D-021, D-022). Results are model output, not field trials.

## 7. Simulator, dashboards, advisory

* **3D simulator** (Three.js, one self-contained page): crop-specific plant models that grow over the season, flooded and terraced paddies, soil that darkens when wet and cracks when dry, daily weather with dates, a time-of-day control, an uncertainty view and a by-stage stress panel; the relief is real (open terrain tiles) but fields, plants and soil classes are schematic. Checked in headless Chromium with software rendering (6 districts, 2772 scenarios), which ran at 13 frames per second; software rendering is far slower than a GPU, so this is a lower bound, not a measurement of the 30 fps goal. Frame rate on real hardware: MISSING (not measured: needs a real browser on real hardware).
* **Dashboards** (Streamlit, a home page and six pages): 7 addresses loaded in a browser with 0 failed checks (no exception boxes, no failed requests); median warm page loads 1.1-1.9 s.
* **Advisory:** fixed templates in English, Hindi, Tamil and Telugu, at most 320 characters; WhatsApp webhook with signature validation; sending is a dry run unless explicitly switched; 7 of 7 bot smoke checks passed. Translation review: MISSING (no native-speaker review done).
* **Container:** 7 of 7 checks passed (serving image 3,525 MB, non-root, health check answers, models run from the demo bundle, memory 172.5MiB / 8.729GiB); the full test suite inside a container built from `requirements.lock`: 327 passed, 9 skipped in 71.21s (0:01:11) (browser tests skip there: no Chromium or node in the image).
* **Automated tests:** 368 (`make test`), plus the browser checks (`make test-sim test-dashboards test-bot`).
* **Deployed URL:** MISSING (deployment is a manual step, see docs/DEPLOY.md).

## 8. The plan's targets against what was measured

| Plan hypothesis | Measured | Verdict |
|---|---|---|
| Yield RMSE 2.9 and R² 0.82 | pooled RMSE 1.04 t/ha, R² 0.91, relative RMSE 40.4% | Not comparable as stated (the plan fixes neither units nor pooling); the honest comparison is against baselines, where there is no clear gain |
| F1 0.75 for extreme weather | drought F1 0.50 (test), 0.70 (validation); heatwave served as climatology | Not reached on held-out years |
| 30% water saving | 61% (normal), 56% (dry) against the 600 mm baseline | Exceeded in the model, at 98% / 96% of baseline yield |

The plan's figures were hypotheses, not results; they are shown here so that nobody mistakes them for achievements.

## 9. Limitations

* Yields end in 2017; nothing is validated on recent seasons. There is no live weather feed: early-warning pages replay observed history.
* District units are reconstructed 1966 parents; results are not village-level.
* Satellite features are district averages of coarse grids (250 m vegetation index, about 9 km soil moisture), so clouds and mixed land use blur them; weather is IMD 1-degree temperature and 0.25-degree rain grids averaged over polygons.
* Translations, advice wording, crop parameters and prices are unreviewed or unverified. Data licences for redistribution (IMD, ICRISAT, geoBoundaries ODbL) must be checked before publishing the demo bundle (`scripts/check_licenses.md`).

## 10. Reproduce

```bash
make setup && make db-up
make pipeline            # etl -> warehouse -> analytics -> models -> optimization -> simulator (hours the first time: IMD download)
make test                # unit tests;  make test-sim test-dashboards test-bot  for the browser and bot checks
python scripts/make_report.py
```

Without the raw data: `python -m config.bundle restore` copies the demo bundle (`data/sample/`) into place and `make dashboards` runs the app.

## 11. Attribution

IMD (gridded rainfall and temperature); NASA Langley Research Center POWER Project; ICRISAT/TCI District Level Database; geoBoundaries (Runfola et al. 2020) with its source data (ODbL); three.js (MIT). See `scripts/check_licenses.md`.
