"""Build the final report, the README results block, the demo script and the CV bullets from ``reports/*.json``.

Every number printed in those documents is read from a result file when this script runs. A number that cannot be
found is written as ``MISSING (reason)`` and listed at the end; nothing is estimated or remembered (HARD RULE 2 in
CLAUDE.md). Wording is fixed here, so the documents say what the measurements say: where the models do not beat a
baseline, the text says so.

Outputs:
* ``reports/final_report.md`` and ``reports/final_report.pdf`` (Markdown rendered to HTML and printed by Chromium);
* the block between ``<!-- RESULTS:START -->`` and ``<!-- RESULTS:END -->`` in ``README.md``;
* ``docs/DEMO_SCRIPT.md`` and ``docs/CV_BULLETS.md``.

Run: ``python scripts/make_report.py [--no-pdf] [--skip-test-count]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.prescriptive.wording import paired_phrase

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LOG = logging.getLogger("make_report")
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PLAN_TARGETS = {"yield_rmse": "2.9", "yield_r2": "0.82", "f1": "0.75", "water_saving": "30%"}


class Facts:
    """Reads result files lazily and renders numbers, remembering which ones were missing."""

    def __init__(self) -> None:
        self.missing: list[str] = []
        self._json: dict[str, Any] = {}

    def json(self, name: str) -> Any:
        """Parsed ``reports/<name>`` (None when the file does not exist)."""
        if name not in self._json:
            path = REPORTS / name
            self._json[name] = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            )
        return self._json[name]

    def csv(self, name: str) -> pd.DataFrame | None:
        """``reports/<name>`` as a DataFrame (None when absent)."""
        path = REPORTS / name
        return pd.read_csv(path) if path.exists() else None

    def show(self, label: str, getter: Callable[[], Any], fmt: str = "{:.2f}") -> str:
        """Format ``getter()``; any lookup failure becomes ``MISSING (label)`` and is recorded."""
        try:
            value = getter()
            if value is None or (isinstance(value, float) and value != value):
                raise KeyError(label)
            return fmt.format(value)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError, ZeroDivisionError):
            self.missing.append(label)
            return f"MISSING ({label})"


def collect(f: Facts, count_tests: bool) -> dict[str, str]:
    """Every figure used in the documents, as ready-to-print strings."""
    q = f.json("data_quality.json")
    yld = f.json("yield_models.json")
    ext = f.json("extreme_weather.json")
    pre = f.json("prescriptive_results.json")
    dash = f.json("dashboard_checks.json")
    bot = f.json("bot_smoke.json")
    sim = f.json("sim_checks.json")
    ctr = f.json("container_checks.json")
    metrics = f.csv("yield_metrics.csv")

    def metric(model: str, col: str, crop: str = "all") -> float:
        row = metrics[
            (metrics["split"] == "test")
            & (metrics["crop"] == crop)
            & (metrics["lead_days"] == 60)
            & (metrics["feature_set"] == "weather")
            & (metrics["model"] == model)
        ]
        return float(row.iloc[0][col])

    def best_baseline() -> str:
        names = ("naive_last_year", "district_trend", "prior_mean_3y")
        return min(names, key=lambda n: metric(n, "rmse_rel"))

    def best_absolute() -> str:
        names = ("naive_last_year", "district_trend", "prior_mean_3y")
        return min(names, key=lambda n: metric(n, "rmse"))

    def boot(baseline: str, field: str) -> float:
        rows = [
            r
            for r in yld["bootstrap_vs_baseline"]
            if r["model"] == "xgboost"
            and r["baseline"] == baseline
            and r["feature_set"] == "weather"
            and r["lead_days"] == 60
        ]
        return float(rows[0][field])

    def summary(scenario: str, strategy: str, field: str) -> float:
        row = [
            r for r in pre["summary"] if r["scenario"] == scenario and r["strategy"] == strategy
        ][0]
        return float(row[field])

    def paired(scenario: str, field: str) -> float:
        return float(
            [r for r in pre["paired_optimized_vs_even"] if r["scenario"] == scenario][0][field]
        )

    def drought(field: str, split: str = "test") -> float:
        return float(
            ext["drought"]["feature_sets"][ext["drought"]["served_feature_set"]][split][field]
        )

    def pooled(forecast: str, field: str) -> float:
        rows = [
            r
            for r in ext["heatwave"]["pooled_metrics"]
            if r["horizon_days"] == 14 and r["forecast"] == forecast
        ]
        return float(rows[0][field])

    def scenario(name: str) -> float:
        rows = [r for r in yld["scenario_response"] if r["scenario"] == name and r["crop"] == "all"]
        return float(rows[0]["mean_change_pct"])

    def warm_median(page: str) -> float:
        return float(dash["pages"][page]["warm_s"])

    def sat_synthetic() -> bool:
        return q["synthetic_share"]["satellite_16d"] > 0

    def drought_uses() -> str:
        served = ext["drought"]["served_feature_set"]
        return "weather" if served == "weather" else "weather and satellite"

    def sat_use() -> str:
        return (
            "the served yield model uses weather features only and the served drought model uses "
            f"{drought_uses()} features"
        )

    def sat_headline() -> str:
        if sat_synthetic():
            return (
                "Satellite NDVI/soil-moisture inputs are **synthetic** (Earth Engine was not "
                "authenticated). The served models do not use them."
            )
        return (
            "Satellite NDVI and soil moisture are **real** (MODIS and ERA5-Land through Google "
            f"Earth Engine); {sat_use()}."
        )

    def sat_limit() -> str:
        if sat_synthetic():
            return "Satellite features are synthetic"
        return (
            "Satellite features are district averages of coarse grids (250 m vegetation index, "
            "about 9 km soil moisture), so clouds and mixed land use blur them"
        )

    def sat_intro() -> str:
        if sat_synthetic():
            return "the satellite table is synthetic and the served models do not use it"
        return f"the satellite data is real; {sat_use()}"

    out: dict[str, str] = {
        "sat_use": f.show("satellite use in served models", sat_use, "{}"),
        "sat_headline": f.show("satellite status sentence", sat_headline, "{}"),
        "sat_limit": f.show("satellite limitation", sat_limit, "{}"),
        "sat_intro": f.show("satellite status in the demo script", sat_intro, "{}"),
        "sat_banner": f.show(
            "banner pointer", lambda: "the yellow banner and " if sat_synthetic() else "", "{}"
        ),
        "sat_next": f.show(
            "next steps prefix", lambda: "real satellite data, " if sat_synthetic() else "", "{}"
        ),
        "n_districts": f.show("districts", lambda: q["rows"]["districts"], "{:d}"),
        "n_weather": f.show("weather rows", lambda: q["rows"]["weather_daily"], "{:,}"),
        "n_yield": f.show("yield rows", lambda: q["rows"]["crops_annual"], "{:,}"),
        "n_sat": f.show("satellite rows", lambda: q["rows"]["satellite_16d"], "{:,}"),
        "sat_synth": f.show(
            "satellite synthetic share", lambda: q["synthetic_share"]["satellite_16d"], "{:.0%}"
        ),
        "dq_ok": "all data-quality checks passed"
        if q and q.get("passed")
        else f.show("data-quality verdict", lambda: None),
        "y_rmse": f.show("xgboost test rmse", lambda: metric("xgboost", "rmse")),
        "y_mae": f.show("xgboost test mae", lambda: metric("xgboost", "mae")),
        "y_r2": f.show("xgboost test r2", lambda: metric("xgboost", "r2")),
        "y_rel": f.show(
            "xgboost test relative rmse", lambda: metric("xgboost", "rmse_rel"), "{:.1%}"
        ),
        "y_best": f.show("best baseline", lambda: best_baseline().replace("_", " "), "{}"),
        "y_best_rel": f.show(
            "best baseline relative rmse", lambda: metric(best_baseline(), "rmse_rel"), "{:.1%}"
        ),
        "y_best_rmse": f.show("best baseline rmse", lambda: metric(best_baseline(), "rmse")),
        "y_abs_best": f.show(
            "best baseline by pooled rmse", lambda: best_absolute().replace("_", " "), "{}"
        ),
        "y_abs_best_rmse": f.show("its pooled rmse", lambda: metric(best_absolute(), "rmse")),
        "y_abs_word": f.show(
            "pooled comparison word",
            lambda: (
                "lower" if metric(best_absolute(), "rmse") < metric("xgboost", "rmse") else "higher"
            ),
            "{}",
        ),
        "y_naive_r2": f.show("naive r2", lambda: metric("naive_last_year", "r2")),
        "y_naive_ratio": f.show("ratio vs naive", lambda: boot("naive_last_year", "ratio")),
        "y_trend_ratio": f.show("ratio vs trend", lambda: boot("district_trend", "ratio")),
        "y_trend_lo": f.show("ratio vs trend low", lambda: boot("district_trend", "lo")),
        "y_trend_hi": f.show("ratio vs trend high", lambda: boot("district_trend", "hi")),
        "y_mean_ratio": f.show("ratio vs 3-year mean", lambda: boot("prior_mean_3y", "ratio")),
        "y_mean_lo": f.show("ratio vs 3-year mean low", lambda: boot("prior_mean_3y", "lo")),
        "y_mean_hi": f.show("ratio vs 3-year mean high", lambda: boot("prior_mean_3y", "hi")),
        "y_cover": f.show(
            "interval coverage", lambda: yld["intervals_test"]["60"]["coverage"], "{:.0%}"
        ),
        "y_rain06": f.show("rain x0.6 response", lambda: scenario("rain x0.6"), "{:+.1f}%"),
        "y_warm4": f.show("+4C response", lambda: scenario("+4 C"), "{:+.1f}%"),
        "d_f1": f.show("drought test F1", lambda: drought("f1")),
        "d_f1_valid": f.show("drought valid F1", lambda: drought("f1", "valid")),
        "d_prec": f.show("drought precision", lambda: drought("precision")),
        "d_rec": f.show("drought recall", lambda: drought("recall")),
        "d_ap": f.show("drought PR-AUC", lambda: drought("pr_auc")),
        "d_ap_clim": f.show(
            "climatology PR-AUC", lambda: ext["drought"]["baselines_test"]["climatology"]["pr_auc"]
        ),
        "d_ap_pers": f.show(
            "persistence PR-AUC", lambda: ext["drought"]["baselines_test"]["persistence"]["pr_auc"]
        ),
        "d_base": f.show("drought base rate", lambda: drought("base_rate"), "{:.1%}"),
        "h_onsets": f.show(
            "heatwave onsets tested", lambda: ext["heatwave"]["pooled_onsets_tested"], "{:d}"
        ),
        "h_served": f.show(
            "heatwave served source", lambda: ext["heatwave"]["served"]["source"], "{}"
        ),
        "h_ap_lstm": f.show("LSTM PR-AUC 14d", lambda: pooled("lstm", "pr_auc"), "{:.3f}"),
        "h_ap_clim": f.show(
            "climatology PR-AUC 14d", lambda: pooled("climatology", "pr_auc"), "{:.3f}"
        ),
        "h_f1_clim": f.show("climatology F1 14d", lambda: pooled("climatology", "f1"), "{:.2f}"),
        "w_saved": f.show(
            "water saved normal",
            lambda: summary("normal rainfall", "optimized", "water_saved_pct"),
            "{:.0f}%",
        ),
        "w_saved_dry": f.show(
            "water saved dry",
            lambda: summary("dry (rain x0.7)", "optimized", "water_saved_pct"),
            "{:.0f}%",
        ),
        "w_yield": f.show(
            "yield vs baseline normal",
            lambda: summary("normal rainfall", "optimized", "yield_vs_baseline"),
            "{:.0%}",
        ),
        "w_yield_dry": f.show(
            "yield vs baseline dry",
            lambda: summary("dry (rain x0.7)", "optimized", "yield_vs_baseline"),
            "{:.0%}",
        ),
        "w_even_water": f.show(
            "water vs even split", lambda: -paired("normal rainfall", "delta_water_mm"), "{:.0f}"
        ),
        "w_even_phrase": f.show(
            "yield verdict vs even split",
            lambda: paired_phrase(
                paired("normal rainfall", "delta_rel_yield"),
                paired("normal rainfall", "delta_rel_yield_lo"),
                paired("normal rainfall", "delta_rel_yield_hi"),
                "yield gain",
                "yield loss",
                "no clear yield difference",
            ),
            "{}",
        ),
        "w_even_dy": f.show(
            "yield difference vs even",
            lambda: paired("normal rainfall", "delta_rel_yield"),
            "{:+.3f}",
        ),
        "w_even_lo": f.show(
            "yield difference low",
            lambda: paired("normal rainfall", "delta_rel_yield_lo"),
            "{:+.4f}",
        ),
        "w_even_hi": f.show(
            "yield difference high",
            lambda: paired("normal rainfall", "delta_rel_yield_hi"),
            "{:+.4f}",
        ),
        "w_worse": f.show(
            "cases optimized worse",
            lambda: paired("normal rainfall", "cases_optimized_worse"),
            "{:.0f}",
        ),
        "w_cases": f.show("cases", lambda: paired("normal rainfall", "cases"), "{:.0f}"),
        "w_secs": f.show("max LP case seconds", lambda: pre["max_case_seconds"], "{:.2f}"),
        "dash_pages": f.show("dashboard pages checked", lambda: len(dash["pages"]), "{:d}"),
        "dash_fail": f.show("dashboard failures", lambda: len(dash["failures"]), "{:d}"),
        "dash_warm_max": f.show(
            "slowest warm page load",
            lambda: max(p["warm_s"] for p in dash["pages"].values()),
            "{:.1f}",
        ),
        "dash_warm_min": f.show(
            "fastest warm page load",
            lambda: min(p["warm_s"] for p in dash["pages"].values()),
            "{:.1f}",
        ),
        "dash_yield": f.show("yield page warm load", lambda: warm_median("yield"), "{:.1f}"),
        "bot_ok": f.show(
            "bot checks passed",
            lambda: sum(1 for k, v in bot.items() if isinstance(v, dict) and v.get("ok")),
            "{:d}",
        ),
        "bot_n": f.show(
            "bot checks",
            lambda: sum(1 for v in bot.values() if isinstance(v, dict) and "ok" in v),
            "{:d}",
        ),
        "ctr_tests": f.show(
            "container test summary",
            lambda: ctr["full_test_suite_in_container"]["detail"]["summary"],
            "{}",
        ),
        "ctr_ok": f.show(
            "container checks passed",
            lambda: sum(1 for v in ctr.values() if isinstance(v, dict) and v.get("ok")),
            "{:d}",
        ),
        "ctr_n": f.show(
            "container checks",
            lambda: sum(1 for v in ctr.values() if isinstance(v, dict) and "ok" in v),
            "{:d}",
        ),
        "ctr_mem": f.show("container memory", lambda: ctr["memory_in_use_after_smoke"], "{}"),
        "ctr_mb": f.show(
            "serving image size", lambda: ctr["build_serving_image"]["detail"]["image_mb"], "{:,}"
        ),
        "sim_fps": f.show(
            "simulator fps (software GL)",
            lambda: sim["average_fps_5s_headless_software_gl"],
            "{:.0f}",
        ),
        "sim_scen": f.show(
            "simulator scenarios", lambda: re.sub(r",\s*built.*$", "", sim["scenarios_chip"]), "{}"
        ),
    }
    out["tests"] = _count_tests() if count_tests else "MISSING (test count skipped)"
    if out["tests"].startswith("MISSING"):
        f.missing.append("test count")
    out["real_gpu_fps"] = "MISSING (not measured: needs a real browser on real hardware)"
    out["deployed_url"] = "MISSING (deployment is a manual step, see docs/DEPLOY.md)"
    out["translation_review"] = "MISSING (no native-speaker review done)"
    return out


def _count_tests() -> str:
    """Number of collected pytest tests (the project's ``addopts`` are overridden so the total line is printed)."""
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
    ]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    match = re.search(r"(\d+) tests? collected", run.stdout)
    return match[1] if match else "MISSING (pytest collection failed)"


# ----------------------------------------------------------------------------------------------------------------
# documents
# ----------------------------------------------------------------------------------------------------------------


def results_table(v: dict[str, str]) -> str:
    """The compact results table shared by README and report."""
    return f"""| Area | Measured result (held-out years) | What it means |
|---|---|---|
| Data | {v["n_districts"]} district units, {v["n_weather"]} daily weather rows, {v["n_yield"]} yield records; satellite table {v["sat_synth"]} synthetic | {v["dq_ok"]}; {v["sat_use"]} |
| Yield forecast (2015-17, 60-day lead) | relative RMSE {v["y_rel"]} (pooled RMSE {v["y_rmse"]} t/ha, R² {v["y_r2"]}); best simple baseline ({v["y_best"]}) {v["y_best_rel"]} | Level with the best baselines: ratio vs district trend {v["y_trend_ratio"]} (95% interval {v["y_trend_lo"]} to {v["y_trend_hi"]}); clearly better than last year's yield ({v["y_naive_ratio"]}). 80% interval covers {v["y_cover"]} of test cases |
| Drought warning (2020-24) | F1 {v["d_f1"]}, PR-AUC {v["d_ap"]} vs climatology {v["d_ap_clim"]} (base rate {v["d_base"]}) | Real skill over climatology and persistence ({v["d_ap_pers"]}); validation F1 was {v["d_f1_valid"]} |
| Heatwave warning (14 days) | LSTM PR-AUC {v["h_ap_lstm"]} vs climatology {v["h_ap_clim"]} on {v["h_onsets"]} pooled onsets | No skill over climatology, so the **{v["h_served"]}** probability is served |
| Irrigation optimizer (normal rain) | saves {v["w_saved"]} of the 600 mm baseline water; keeps {v["w_yield"]} of baseline yield | Modelled, not field-tested. Vs an even split: {v["w_even_water"]} mm less water, yield difference {v["w_even_dy"]} (interval {v["w_even_lo"]} to {v["w_even_hi"]}), worse in {v["w_worse"]} of {v["w_cases"]} cases |
| Dashboards | {v["dash_pages"]} addresses, {v["dash_fail"]} failed browser checks, median warm page load {v["dash_warm_min"]}-{v["dash_warm_max"]} s | Headless Chromium with software rendering |
| Advisory bot | {v["bot_ok"]} of {v["bot_n"]} smoke checks passed | Dry run by default; translations unreviewed |
| Tests | {v["tests"]} automated tests | `make test`; in a clean Linux container: {v["ctr_tests"]} |
"""


def final_report(v: dict[str, str]) -> str:
    """The full report (Markdown)."""
    findings = _findings_block()
    now = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"""# ClimateResilience-India: final report

*Generated {now} by `scripts/make_report.py`. Every number below is read from `reports/*.json` and `reports/*.csv`; a number that could not be found is printed as `MISSING (...)`.*

## 1. Summary

ClimateResilience-India is a district-level decision-support system for four southern states (Andhra Pradesh, Telangana, Karnataka, Tamil Nadu). It ingests weather and crop-yield data, stores them in a PostGIS warehouse, analyses climate risk, forecasts yields, warns about heatwaves and droughts, optimizes irrigation, shows the season in a 3D simulator and delivers advice through Streamlit dashboards and a WhatsApp/SMS assistant.

**What the measurements say, plainly:**

{results_table(v)}

* The yield models are **not better than simple baselines** by a margin the data can confirm. They are useful as a weather-adjusted baseline with an honest interval, not as precise predictors.
* The heatwave LSTM has **no skill over climatology**; the system serves climatology and says so. The drought model **does** beat climatology and persistence.
* The optimizer **saves a large share of the water but keeps less yield than the 600 mm baseline** in the model. Against an even split of the same kind of budget it uses less water with {v["w_even_phrase"]}. The plan's story that yield recovers while water is saved is not what the models show.
* {v["sat_headline"]}

## 2. Data and scope

Units are the {v["n_districts"]} yield-reporting district groups that ICRISAT publishes (1966 boundaries, reconstructed from 2021 polygons), not villages. Sources: IMD gridded rainfall and temperature, NASA POWER (humidity, wind, solar), ICRISAT district yields (2000-2017), geoBoundaries polygons. Rows: {v["n_weather"]} daily weather, {v["n_yield"]} yield records, {v["n_sat"]} satellite rows ({v["sat_synth"]} synthetic, flagged `is_synthetic`). Data-quality verdict: {v["dq_ok"]} (`docs/DATA_QUALITY.md`). Decisions that shape the results are logged in `docs/DECISIONS.md`.

## 3. Climate findings (descriptive and diagnostic)

{findings}

## 4. Yield forecasting

Target: yield relative to the district's prior mean. Time-based splits only (train to 2011, validation 2012-14, test 2015-17). Models: naive last year, district trend, prior-three-year mean, Ridge, random forest, XGBoost (Optuna, monotone temperature constraint), LSTM; 80% intervals by conformalized quantile regression.

* Test relative RMSE of the served XGBoost: {v["y_rel"]}; best simple baseline ({v["y_best"]}): {v["y_best_rel"]}. In absolute terms the pooled RMSE is {v["y_rmse"]} t/ha against {v["y_best_rmse"]} t/ha for that baseline; the best baseline by pooled RMSE is the {v["y_abs_best"]} ({v["y_abs_best_rmse"]} t/ha, {v["y_abs_word"]} than the model's). Pooled errors are dominated by sugarcane's high yields, and R² ({v["y_r2"]}) says little: last year's yield alone already reaches {v["y_naive_r2"]}, because yields differ strongly between districts.
* Cluster-bootstrap ratios of relative RMSE (below 1 is better): vs last year {v["y_naive_ratio"]}; vs district trend {v["y_trend_ratio"]} ({v["y_trend_lo"]} to {v["y_trend_hi"]}); vs prior three-year mean {v["y_mean_ratio"]} ({v["y_mean_lo"]} to {v["y_mean_hi"]}).
* 80% interval coverage on the test years: {v["y_cover"]}.
* What-if sensitivity of the served model (validation years): {v["y_rain06"]} for 40% less rain, {v["y_warm4"]} for +4 °C: the model learned yield mostly from each district's own history.
* Full tables and SHAP: `reports/yield_model_comparison.md`, `reports/yield_metrics.csv`, `reports/figures/yield_shap_*.png`.

## 5. Heatwave and drought early warning

* **Drought** (SPI-3 at or below -1 next month; XGBoost on rainfall-deficit features, isotonic calibration): test F1 {v["d_f1"]} (precision {v["d_prec"]}, recall {v["d_rec"]}), PR-AUC {v["d_ap"]} against {v["d_ap_clim"]} for climatology and {v["d_ap_pers"]} for persistence; validation F1 {v["d_f1_valid"]}. A drought signal for a normally dry month does not set the risk level (D-024).
* **Heatwave** (onset within 14 days, March-June): pooled rolling-origin PR-AUC {v["h_ap_lstm"]} for the LSTM against {v["h_ap_clim"]} for climatology over {v["h_onsets"]} onsets; the served probability is **{v["h_served"]}**. Onsets are rare, so conclusions are weak in both directions.
* Details: `reports/extreme_weather_report.md`.

## 6. Irrigation optimization

Daily FAO-56 water balance, FAO-33 yield response and a weekly linear programme (PuLP/CBC); the slowest LP case took {v["w_secs"]} s. Paddy rice is modelled as a ponded field (standing water behind bunds and seepage while it stands; the parameters are approximate and unverified, D-029); the other crops use the plain root-zone balance.

* Normal rainfall: {v["w_saved"]} water saved against the 600 mm baseline, yield {v["w_yield"]} of baseline. Dry year (rain x0.7): {v["w_saved_dry"]} saved, yield {v["w_yield_dry"]} of baseline.
* Against an even split: {v["w_even_water"]} mm less water, relative-yield difference {v["w_even_dy"]} (interval {v["w_even_lo"]} to {v["w_even_hi"]}), worse in {v["w_worse"]} of {v["w_cases"]} cases.
* Crop parameters and prices are approximate and unverified (D-021, D-022). Results are model output, not field trials.

## 7. Simulator, dashboards, advisory

* **3D simulator** (Three.js, one self-contained page): crop-specific plant models that grow over the season, flooded and terraced paddies, soil that darkens when wet and cracks when dry, daily weather with dates, a time-of-day control, an uncertainty view and a by-stage stress panel; the relief is real (open terrain tiles) but fields, plants and soil classes are schematic. Checked in headless Chromium with software rendering ({v["sim_scen"]}), which ran at {v["sim_fps"]} frames per second; software rendering is far slower than a GPU, so this is a lower bound, not a measurement of the 30 fps goal. Frame rate on real hardware: {v["real_gpu_fps"]}.
* **Dashboards** (Streamlit, a home page and six pages): {v["dash_pages"]} addresses loaded in a browser with {v["dash_fail"]} failed checks (no exception boxes, no failed requests); median warm page loads {v["dash_warm_min"]}-{v["dash_warm_max"]} s.
* **Advisory:** fixed templates in English, Hindi, Tamil and Telugu, at most 320 characters; WhatsApp webhook with signature validation; sending is a dry run unless explicitly switched; {v["bot_ok"]} of {v["bot_n"]} bot smoke checks passed. Translation review: {v["translation_review"]}.
* **Container:** {v["ctr_ok"]} of {v["ctr_n"]} checks passed (serving image {v["ctr_mb"]} MB, non-root, health check answers, models run from the demo bundle, memory {v["ctr_mem"]}); the full test suite inside a container built from `requirements.lock`: {v["ctr_tests"]} (browser tests skip there: no Chromium or node in the image).
* **Automated tests:** {v["tests"]} (`make test`), plus the browser checks (`make test-sim test-dashboards test-bot`).
* **Deployed URL:** {v["deployed_url"]}.

## 8. The plan's targets against what was measured

| Plan hypothesis | Measured | Verdict |
|---|---|---|
| Yield RMSE {PLAN_TARGETS["yield_rmse"]} and R² {PLAN_TARGETS["yield_r2"]} | pooled RMSE {v["y_rmse"]} t/ha, R² {v["y_r2"]}, relative RMSE {v["y_rel"]} | Not comparable as stated (the plan fixes neither units nor pooling); the honest comparison is against baselines, where there is no clear gain |
| F1 {PLAN_TARGETS["f1"]} for extreme weather | drought F1 {v["d_f1"]} (test), {v["d_f1_valid"]} (validation); heatwave served as climatology | Not reached on held-out years |
| {PLAN_TARGETS["water_saving"]} water saving | {v["w_saved"]} (normal), {v["w_saved_dry"]} (dry) against the 600 mm baseline | Exceeded in the model, at {v["w_yield"]} / {v["w_yield_dry"]} of baseline yield |

The plan's figures were hypotheses, not results; they are shown here so that nobody mistakes them for achievements.

## 9. Limitations

* Yields end in 2017; nothing is validated on recent seasons. There is no live weather feed: early-warning pages replay observed history.
* District units are reconstructed 1966 parents; results are not village-level.
* {v["sat_limit"]}; weather is IMD 1-degree temperature and 0.25-degree rain grids averaged over polygons.
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
"""


def _findings_block() -> str:
    """The five headline findings copied from the generated ``docs/FINDINGS.md`` (numbers computed there)."""
    path = ROOT / "docs" / "FINDINGS.md"
    if not path.exists():
        return "MISSING (docs/FINDINGS.md: run `make analytics-eda`)"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## Five most important findings\n\n(.*?)\n## ", text, re.S)
    if not match:
        return "MISSING (findings section not found in docs/FINDINGS.md)"
    return match[1].strip() + "\n\nFull tables: `docs/FINDINGS.md`, `reports/analytics/`."


def readme_block(v: dict[str, str]) -> str:
    """Auto-generated block of README.md."""
    return f"{START}\n_Generated by `python scripts/make_report.py` from `reports/*.json`; do not edit by hand._\n\n{results_table(v)}\n{END}"


def demo_script(v: dict[str, str]) -> str:
    """A five-minute demonstration that follows the measured results."""
    return f"""# Demo script (about 5 minutes)

*Generated by `scripts/make_report.py`; the numbers are the measured ones. Say the cautions out loud: they are part of the story. Start with `make dashboards` (or `docker compose up app`) and open http://localhost:8501.*

**0:00 Home.** "This covers {v["n_districts"]} district units in four southern states, built from IMD weather and ICRISAT yields. Two things to know up front: {v["sat_intro"]}, and there is no live weather feed, so everything replays observed history." Point at {v["sat_banner"]}the **Measured results** tiles.

**0:45 Weather and early warning** (district Thanjavur, date 2024-06-15, then move to a drought year such as 2003-06-15 for Warangal). "Two hazards. Drought next month uses a model that beats climatology: PR-AUC {v["d_ap"]} against {v["d_ap_clim"]}, F1 {v["d_f1"]} on 2020-24. Heatwave uses plain climatology, because my LSTM did not beat it ({v["h_ap_lstm"]} against {v["h_ap_clim"]} PR-AUC): I serve what works." Show the probability bars and the 60-day weather.

**1:45 Risk map.** Switch between drought and heatwave, then the long-term vulnerability tab. "Colour is probability on the chosen date; the second tab ranks districts by drought frequency, yield variability and heat days. It describes history, it is not a forecast."

**2:30 Yield predictor.** Rice, Thanjavur. Show the forecast with its 80% interval ({v["y_cover"]} of test cases fall inside) and the SHAP drivers. Move the rainfall slider. "Honest finding: the forecast hardly moves ({v["y_rain06"]} for 40% less rain). The model mostly learned each district's own history. Against the best simple baseline its error is {v["y_rel"]} against {v["y_best_rel"]} of the mean yield: no clear gain, so I show an interval, not a promise." Open the crop comparison and point out when the pessimistic ranking differs.

**3:30 3D simulator.** Choose rice, step through the weeks (seedlings, then tufts, then heads, with the date of the weather shown) and use Compare: the farmer-baseline paddy stays flooded while the optimized one is drained between irrigation events. Press U for the uncertainty view (three strips: low end, model, high end of the 80% interval) and read the water-stress-by-stage panel. "The optimizer saves {v["w_saved"]} of the water in the model but keeps {v["w_yield"]} of the baseline yield; against an even split it saves {v["w_even_water"]} mm with {v["w_even_phrase"]}. That is a trade-off for agronomists to judge, not a free lunch." Mention that the frame rate was only checked with software rendering ({v["sim_fps"]} fps); real-GPU performance: {v["real_gpu_fps"]}.

**4:15 Farmer advisory.** Show the same district in English and Tamil. "Fixed templates, no language model, 320 characters; the WhatsApp bot asks before using an uncertain district name and honours STOP. Sending is a dry run by default. The translations are drafts: {v["translation_review"]}."

**4:50 Methodology and close.** "Everything on this page is read from the report files. Tests: {v["tests"]}; browser checks: {v["dash_pages"]} dashboard addresses with {v["dash_fail"]} failures. Next steps: {v["sat_next"]}a live forecast feed, reviewed translations, field validation."

## Questions to be ready for

* *Why not village level?* Yields exist only for ICRISAT's 1966 districts.
* *Why is the heatwave model not used?* It had no skill over climatology on {v["h_onsets"]} pooled onsets (only a handful per fold).
* *Is the water saving real?* It is a model result with approximate crop parameters; no field trial.
* *Can I use it with farmers?* Not before translation and agronomy review, and a licence check for the data (`scripts/check_licenses.md`).
"""


def cv_bullets(v: dict[str, str]) -> str:
    """CV and portfolio bullets with measured numbers only."""
    return f"""# CV bullets (measured numbers only)

*Generated by `scripts/make_report.py`. Use a bullet only if you can defend the number; items marked MISSING were not measured.*

## Short

* Built an end-to-end climate-risk platform for {v["n_districts"]} district units in four Indian states: ETL of IMD, NASA POWER and ICRISAT data ({v["n_weather"]} daily weather rows, {v["n_yield"]} yield records) into a PostGIS warehouse, ML models, an LP irrigation optimizer, a Three.js simulator, Streamlit dashboards and a WhatsApp advisory bot.
* Developed a drought early-warning model (XGBoost with isotonic calibration, time-based split) that reaches PR-AUC {v["d_ap"]} against {v["d_ap_clim"]} for climatology on held-out 2020-24 data (F1 {v["d_f1"]}); showed with rolling-origin evaluation that an LSTM heatwave model had no skill over climatology and served the baseline instead.
* Evaluated crop-yield models with leakage-safe features, cluster-bootstrap confidence intervals and conformal 80% intervals ({v["y_cover"]} test coverage); reported honestly that they match but do not clearly beat district-trend and prior-mean baselines (relative RMSE {v["y_rel"]} vs {v["y_best_rel"]}).
* Formulated a weekly irrigation LP on an FAO-56 water balance that cuts modelled water use by {v["w_saved"]} against a 600 mm baseline (slowest case {v["w_secs"]} s), with the yield trade-off ({v["w_yield"]} of baseline) reported alongside.
* Engineered for safety and reproducibility: {v["tests"]} automated tests, browser tests of the dashboards and simulator, signed WhatsApp webhook, dry-run-by-default messaging with rate limits and opt-out, seeded runs, Docker image, one-command pipeline.

## Longer (for a project page)

* Data engineering: idempotent ETL with provenance flags (`is_synthetic`), data-quality gate ({v["dq_ok"]}), warehouse with leakage-safe materialized views.
* Statistics: two-way fixed-effects regression, Mann-Kendall trends, SPI drought index, calibrated probabilities with reliability checks, paired bootstrap against baselines.
* Front ends: Streamlit app with {v["dash_pages"]} addresses (warm loads {v["dash_warm_min"]}-{v["dash_warm_max"]} s in headless Chromium), self-contained Three.js scene.

## Not claimable (MISSING)

* Real-GPU frame rate of the simulator: {v["real_gpu_fps"]}.
* Deployed public URL: {v["deployed_url"]}.
* Native-speaker review of the Hindi, Tamil and Telugu advisories: {v["translation_review"]}.
* Any farmer or agronomist feedback or field validation: MISSING (not done).
"""


def verify_claims(f: Facts) -> list[str]:
    """Qualitative sentences of the documents that must still be true; returns the ones the numbers contradict.

    The prose in this file says, for example, that the heatwave LSTM has no skill and that the optimizer keeps less
    yield than the baseline. If a re-run changes that, the documents must be rewritten, not silently left wrong.
    """
    failed: list[str] = []
    try:
        yld, ext, pre, qual = (
            f.json(n)
            for n in (
                "yield_models.json",
                "extreme_weather.json",
                "prescriptive_results.json",
                "data_quality.json",
            )
        )
        rows = {
            (r["model"], r["baseline"]): r
            for r in yld["bootstrap_vs_baseline"]
            if r["feature_set"] == "weather" and r["lead_days"] == 60
        }
        xgb = lambda base: rows[("xgboost", base)]  # noqa: E731
        if not xgb("naive_last_year")["hi"] < 1.0:
            failed.append("the yield model no longer clearly beats last year's yield")
        for base in ("district_trend", "prior_mean_3y"):
            if xgb(base)["hi"] < 1.0:
                failed.append(
                    f"the yield model now clearly beats the {base} baseline: reword 'not clearly better'"
                )
        heat = {
            r["forecast"]: r["pr_auc"]
            for r in ext["heatwave"]["pooled_metrics"]
            if r["horizon_days"] == 14
        }
        if (
            ext["heatwave"]["served"]["source"] != "climatology"
            or heat["lstm"] >= heat["climatology"]
        ):
            failed.append(
                "the heatwave LSTM now matches climatology: reword the heatwave statements"
            )
        fs = ext["drought"]["feature_sets"][ext["drought"]["served_feature_set"]]["test"]
        if fs["pr_auc"] <= max(
            ext["drought"]["baselines_test"][b]["pr_auc"] for b in ("climatology", "persistence")
        ):
            failed.append("the drought model no longer beats climatology and persistence")
        normal = [
            r
            for r in pre["summary"]
            if r["scenario"] == "normal rainfall" and r["strategy"] == "optimized"
        ][0]
        if not normal["yield_vs_baseline"] < 1.0 or normal["water_saved_pct"] <= 30:
            failed.append(
                "the optimizer no longer saves water at a yield cost against the baseline"
            )
    except (KeyError, IndexError, TypeError):
        failed.append("result files are missing or have an unexpected structure")
    return failed


def write_readme_block(block: str) -> bool:
    """Insert or replace the results block in README.md; False if the markers are absent."""
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    path.write_text(new, encoding="utf-8")
    return True


def write_pdf(markdown_text: str, target: Path) -> None:
    """Render Markdown to HTML and print it to PDF with headless Chromium (A4)."""
    import markdown
    from playwright.sync_api import sync_playwright

    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code", "sane_lists"])
    style = """
      body{font-family:'Segoe UI','DejaVu Sans',Arial,sans-serif;font-size:10.5pt;line-height:1.45;color:#0b0b0b;max-width:none}
      h1,h2{break-after:avoid} h1{font-size:20pt;margin:0 0 .4em} h2{font-size:14pt;margin:1.3em 0 .4em;border-bottom:1px solid #c3c2b7;padding-bottom:.15em}
      table{border-collapse:collapse;width:100%;margin:.6em 0;font-size:9pt} th,td{border:1px solid #c3c2b7;padding:3px 6px;vertical-align:top;text-align:left}
      th{background:#f0efec} code{background:#f0efec;padding:0 3px;border-radius:3px;font-size:9pt} pre{background:#f0efec;padding:8px;overflow:auto}
      em{color:#52514e} li{margin:.15em 0}
    """
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>ClimateResilience-India final report</title><style>{style}</style></head><body>{body}</body></html>"
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        page.set_content(html)
        page.pdf(
            path=str(target),
            format="A4",
            margin={"top": "18mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
            print_background=True,
        )
        browser.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit code 1 when any figure is MISSING (the documents are still written)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--skip-test-count", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    facts = Facts()
    values = collect(facts, count_tests=not args.skip_test_count)
    contradictions = verify_claims(facts)
    for claim in contradictions:
        LOG.error("CLAIM NOT SUPPORTED: %s", claim)
    report = final_report(values)
    (REPORTS / "final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "docs" / "DEMO_SCRIPT.md").write_text(demo_script(values), encoding="utf-8")
    (ROOT / "docs" / "CV_BULLETS.md").write_text(cv_bullets(values), encoding="utf-8")
    LOG.info("wrote reports/final_report.md, docs/DEMO_SCRIPT.md, docs/CV_BULLETS.md")
    if write_readme_block(readme_block(values)):
        LOG.info("updated the results block of README.md")
    else:
        LOG.warning("README.md has no %s markers: block not written", START)
    if not args.no_pdf:
        write_pdf(report, REPORTS / "final_report.pdf")
        LOG.info("wrote reports/final_report.pdf")
    expected_missing = {"test count"} if args.skip_test_count else set()
    unexpected = sorted(set(facts.missing) - expected_missing)
    for item in unexpected:
        LOG.warning("MISSING: %s", item)
    return 1 if unexpected or contradictions else 0


if __name__ == "__main__":
    sys.exit(main())
